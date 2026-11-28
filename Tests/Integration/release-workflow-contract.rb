#!/usr/bin/env ruby
# Checks the release job graph: publishing waits for a build and for verification of the same packaged
# artifact on the deployment target (macOS 15), and nothing after the build rebuilds the binary.
require "yaml"
require "minitest/autorun"

WORKFLOW = YAML.load_file(ARGV.shift || ".github/workflows/release.yml")
CI_WORKFLOW = YAML.load_file(ARGV.shift || ".github/workflows/ci.yml")

class ReleaseWorkflowContract < Minitest::Test
  def setup
    @jobs = WORKFLOW.fetch("jobs")
    @publisher_id, @publisher = @jobs.find do |_, job|
      job.fetch("steps").any? { |step| step.fetch("uses", "").start_with?("softprops/action-gh-release@") }
    end
    refute_nil @publisher, "the workflow must publish a release"
  end

  def test_publish_waits_for_build_and_macos_15_verification
    build = @jobs.fetch("build")
    verification = @jobs.fetch("verify-macos-15")
    assert_equal "xcode-27", build.fetch("runs-on")
    assert_equal "macos-15", verification.fetch("runs-on")
    assert_includes Array(verification.fetch("needs")), "build"
    assert_includes Array(@publisher.fetch("needs")), "build"
    assert_includes Array(@publisher.fetch("needs")), "verify-macos-15"
    refute @publisher.key?("if"), "publish must not bypass a failed verification"
    refute verification.key?("if"), "verification must not bypass a failed build"
    assert_equal "read", WORKFLOW.fetch("permissions").fetch("contents")
    assert_equal "write", @publisher.fetch("permissions").fetch("contents")
  end

  def test_verification_and_publish_consume_the_same_packaged_artifact
    upload = @jobs.fetch("build").fetch("steps").find { |step| step.fetch("uses", "").start_with?("actions/upload-artifact@") }
    refute_nil upload
    artifact_name = upload.fetch("with").fetch("name")
    [@jobs.fetch("verify-macos-15"), @publisher].each do |job|
      download = job.fetch("steps").find { |step| step.fetch("uses", "").start_with?("actions/download-artifact@") }
      refute_nil download
      assert_equal artifact_name, download.fetch("with").fetch("name")
      refute job.fetch("steps").any? { |step| step.fetch("run", "").include?("swift build") },
             "verification and publishing must not rebuild the binary"
    end
    commands = @jobs.fetch("verify-macos-15").fetch("steps").map { |step| step.fetch("run", "") }.join("\n")
    assert_includes commands, "shasum -a 256 -c"
    assert_includes commands, "tar -xzf"
    assert_includes commands, "codesign --verify --strict"
    assert_includes commands, "otool -l verify/anerun"
    assert_includes commands, "LC_LOAD_WEAK_DYLIB"
    assert_includes commands, "ANERUN_CHECK_RELEASE_COREAI=1 Tests/Integration/release-binary-launch-smoke.sh verify/anerun"
  end

  def test_one_asset_per_platform
    files = @publisher.fetch("steps").find { |step| step.fetch("uses", "").start_with?("softprops/action-gh-release@") }
                      .fetch("with").fetch("files")
    assert_equal ["dist/*.tar.gz", "dist/*.tar.gz.sha256"], files.split("\n").map(&:strip).reject(&:empty?)
    build = @jobs.fetch("build").fetch("steps").map { |step| step.fetch("run", "") }.join("\n")
    assert_includes build, "-DANERUN_REQUIRE_COREAI"
    assert_includes build, "CoreAI.framework"
    assert_includes build, 'TARBALL="anerun-${GITHUB_REF_NAME}-macos-arm64.tar.gz"'
  end

  def test_third_party_actions_are_pinned_to_commits
    [WORKFLOW, CI_WORKFLOW].each do |workflow|
      workflow.fetch("jobs").each do |name, job|
        job.fetch("steps").each do |step|
          next unless step.key?("uses")
          assert_match(/\A[\w.-]+\/[\w.-]+@[0-9a-f]{40}\z/, step.fetch("uses"), "#{name}: #{step.fetch("uses")} is not pinned to a commit")
        end
      end
    end
  end

  def test_least_privilege_and_one_release_at_a_time
    assert_equal({ "contents" => "read" }, CI_WORKFLOW.fetch("permissions"))
    assert_equal({ "contents" => "read" }, WORKFLOW.fetch("permissions"))
    @jobs.each do |name, job|
      next if job.equal?(@publisher)
      refute job.key?("permissions"), "#{name} must keep the read-only default"
    end
    concurrency = WORKFLOW.fetch("concurrency")
    assert_includes concurrency.fetch("group"), "${{ github.ref }}"
    assert_equal false, concurrency.fetch("cancel-in-progress")
  end

  def test_unit_tests_run_before_packaging
    steps = @jobs.fetch("build").fetch("steps").map { |step| step.fetch("run", "") }
    test_index = steps.index { |run| run.match?(/\bswift test\b/) }
    package_index = steps.index { |run| run.include?("scripts/package-rc.sh") }
    refute_nil test_index, "the release build must run the unit tests"
    refute_nil package_index
    assert_operator test_index, :<, package_index
  end
end
