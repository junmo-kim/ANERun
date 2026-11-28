"""Exercise PPLX cache selection and stage reuse with filesystem-only fixtures."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pplx_build as build


SOURCE = {"id": "owner/model", "revision": "a" * 40, "sha256": {"weights": "b" * 64}}
SPEC = {"function": "pre0", "package": "pre0", "role": "pre", "inputs": [{"shape": [1, 2]}]}


def compiled_fixture(path):
    (path / "weights").mkdir(parents=True)
    for file in ("model.mil", "coremldata.bin", "weights/weight.bin"):
        (path / file).write_bytes(b"fixture")


class PPLXCacheTests(unittest.TestCase):
    def test_key_tracks_sources_topology_pin_and_dependencies(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for name in ("build.py", "pplx_build.py", "pplx.py", "manifest.py", "requirements.txt"):
                (directory / name).write_text(name)
            with patch.object(build, "__file__", str(directory / "pplx_build.py")), patch.object(build, "version") as version:
                version.return_value = "1"
                key = build.trace_key(SOURCE, [SPEC])
                self.assertEqual(key, build.trace_key(dict(reversed(list(SOURCE.items()))), [SPEC]))
                self.assertNotEqual(key, build.trace_key({**SOURCE, "revision": "c" * 40}, [SPEC]))
                self.assertNotEqual(key, build.trace_key({**SOURCE, "sha256": {"weights": "d" * 64}}, [SPEC]))
                self.assertNotEqual(key, build.trace_key(SOURCE, [{**SPEC, "inputs": [{"shape": [1, 3]}]}]))
                version.side_effect = lambda name: "2" if name == "torch" else "1"
                self.assertNotEqual(key, build.trace_key(SOURCE, [SPEC]))
                version.side_effect = None
                (directory / "pplx.py").write_text("different implementation with the same ABI")
                self.assertNotEqual(key, build.trace_key(SOURCE, [SPEC]))

    def test_stages_skip_only_proven_traces_and_refresh_output(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            work = directory / "work-pplx"
            out = directory / "first"
            legacy = work / "fn" / "pre0.mlpackage"
            legacy.mkdir(parents=True)
            (legacy / "trace").write_text("old")
            compiler_path = directory.resolve() / "coremlcompiler"
            compiler_path.write_text("compiler version A")
            args = SimpleNamespace(model=build.MODEL, work=work, out=out, hf_dir=directory / "snapshot", stages="merge", force=False)

            def convert(_, spec, path, *, causal):
                self.assertFalse(causal)
                self.assertEqual(spec, SPEC)
                path.mkdir()
                (path / "trace").write_text(str(conversion.call_count))

            def compile_package(command):
                self.assertEqual(command[:2], [str(compiler_path), "compile"])
                source, staging = map(Path, command[-2:])
                target = staging / source.with_suffix(".mlmodelc").name
                compiled_fixture(target)
                (target / "trace").write_text((source / "trace").read_text())

            compiler = Mock(side_effect=compile_package)

            def run_command(command, *, check, capture_output=False, text=False):
                self.assertTrue(check)
                if command == ["xcrun", "--find", "coremlcompiler"]:
                    self.assertTrue(capture_output and text)
                    return SimpleNamespace(stdout=f"{compiler_path}\n")
                return compiler(command)

            stubs = {"torch": SimpleNamespace(set_num_threads=Mock()), "pplx": SimpleNamespace(Weights=Mock()),
                     "coremltools": SimpleNamespace(), "manifest": SimpleNamespace(check_pplx_spec=Mock())}
            with (
                patch.dict(sys.modules, stubs),
                patch.object(build, "check_source", return_value=SOURCE),
                patch.object(build, "function_specs", side_effect=lambda **_: iter([SPEC])),
                patch.object(build, "trace_key", return_value="first-key") as key,
                patch.object(build, "convert_one", side_effect=convert) as conversion,
                patch.object(build.subprocess, "run", side_effect=run_command),
            ):
                with self.assertRaisesRegex(SystemExit, "run --stages convert first"):
                    build.run(args)
                self.assertFalse((out / "pre0.mlpackage").exists())

                args.stages = "convert"
                build.run(args)
                args.stages = "merge,compile"
                build.run(args)
                self.assertEqual((out / "pre0.mlmodelc" / "trace").read_text(), "1")

                args.out = directory / "sibling"
                args.stages = "convert,merge,compile"
                build.run(args)
                self.assertEqual((args.out / "pre0.mlpackage" / "trace").read_text(), "1")
                self.assertEqual((legacy / "trace").read_text(), "old")
                conversion.assert_called_once()
                self.assertEqual(compiler.call_count, 2)
                build.run(args)
                self.assertEqual(compiler.call_count, 2)

                previous = (args.out / build.COMPILE_KEY).read_text()
                compiler_path.write_text("compiler version B")
                args.stages = "manifest"
                with self.assertRaisesRegex(SystemExit, "compiled PPLX packages are stale"):
                    build.run(args)
                args.stages = "compile"
                build.run(args)
                conversion.assert_called_once()
                self.assertEqual(compiler.call_count, 3)
                self.assertNotEqual((args.out / build.COMPILE_KEY).read_text(), previous)

                key.return_value = "second-key"
                args.stages = "convert,merge"
                build.run(args)
                args.stages = "manifest"
                with self.assertRaisesRegex(SystemExit, "compiled PPLX packages are stale"):
                    build.run(args)
                args.stages = "compile"
                build.run(args)
                self.assertEqual((args.out / "pre0.mlmodelc" / "trace").read_text(), "2")
                args.force = True
                args.stages = "convert,merge,compile"
                build.run(args)
                self.assertEqual((args.out / "pre0.mlmodelc" / "trace").read_text(), "3")
                self.assertEqual(compiler.call_count, 5)

                args.force = False
                args.stages = "compile"
                (args.out / build.MERGE_KEY).write_text(build.MERGE_INCOMPLETE)
                with self.assertRaisesRegex(SystemExit, "merge was interrupted"):
                    build.run(args)
                self.assertEqual(compiler.call_count, 5)

    def test_matching_compile_key_does_not_hide_incomplete_outputs(self):
        for model, package in ((build.MODEL, "pre0"), (build.QWEN, "pre_qwen")):
            with self.subTest(model=model):
                self.check_incomplete_compiler_cache(model, package)

    def check_incomplete_compiler_cache(self, model, package):
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            target = out / f"{package}.mlmodelc"
            target.mkdir()
            (out / build.MERGE_KEY).write_text("graph")
            (out / build.COMPILE_KEY).write_text("compile")
            args = SimpleNamespace(model=model, out=out, hf_dir=out, stages="manifest", force=False)
            manifest = Mock()

            def compile_package(command, *, check):
                self.assertTrue(check)
                self.assertEqual(Path(command[-2]).name, f"{package}.mlpackage")
                compiled_fixture(Path(command[-1]) / target.name)

            with (
                patch.object(build, "check_source", return_value=SOURCE),
                patch.object(build, "function_specs", side_effect=lambda *, causal: iter([
                    {**SPEC, "package": "pre_qwen" if causal else "pre0"},
                ])),
                patch.object(build, "trace_key", return_value="graph"),
                patch.object(build, "select_compiler", return_value=("compiler", "compile")),
                patch.object(build.subprocess, "run", side_effect=compile_package) as compiler,
                patch.dict(sys.modules, {"manifest": SimpleNamespace(write_pplx=manifest)}),
            ):
                with self.assertRaisesRegex(SystemExit, "incomplete compiled PPLX package"):
                    build.run(args)
                manifest.assert_not_called()
                args.stages = "compile,manifest"
                build.run(args)
                compiler.assert_called_once()
                manifest.assert_called_once_with(out, SOURCE)
                (target / "weights/weight.bin").write_bytes(b"")
                args.stages = "manifest"
                with self.assertRaisesRegex(SystemExit, "incomplete compiled PPLX package"):
                    build.run(args)

    def test_stage_only_graph_provenance_is_checked_before_using_or_mutating_output(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            args = SimpleNamespace(model=build.MODEL, out=out, hf_dir=Path(tmp) / "snapshot", stages="assets,selftest", force=False)
            source = {**SOURCE, "revision": "c" * 40}
            selftest = Mock()
            manifest = Mock()
            with (
                patch.object(build, "check_source", return_value=source),
                patch.object(build, "function_specs", side_effect=lambda **_: iter([SPEC])),
                patch.object(build, "trace_key", return_value="current") as key,
                patch.object(build, "stage_assets") as assets,
                patch.object(build.subprocess, "run") as subprocess_run,
                patch.dict(sys.modules, {"selftest": SimpleNamespace(generate_pplx=selftest),
                                         "manifest": SimpleNamespace(write_pplx=manifest)}),
            ):
                build.run(args)
                assets.assert_called_once_with(args)
                selftest.assert_called_once_with(args.hf_dir, out)
                key.assert_not_called()
                assets.reset_mock()
                selftest.reset_mock()

                (out / "pre0.mlpackage").mkdir()
                for existing in (None, "previous"):
                    if existing is not None:
                        (out / build.MERGE_KEY).write_text(existing)
                    for stages in ("assets", "selftest", "compile", "manifest", "assets,selftest,manifest"):
                        with self.subTest(existing=existing, stages=stages):
                            args.stages = stages
                            with self.assertRaisesRegex(SystemExit, "PPLX graph provenance is missing or stale"):
                                build.run(args)
                            key.assert_called_with(source, [SPEC])
                assets.assert_not_called()
                selftest.assert_not_called()
                manifest.assert_not_called()
                subprocess_run.assert_not_called()

                (out / build.MERGE_KEY).write_text("current")
                args.stages = "assets,selftest,manifest"
                build.run(args)
                assets.assert_called_once_with(args)
                selftest.assert_called_once_with(args.hf_dir, out)
                manifest.assert_called_once_with(out, source)
                subprocess_run.assert_not_called()

    def test_compile_key_tracks_toolchain_metadata_and_host(self):
        with TemporaryDirectory() as tmp:
            contents = Path(tmp).resolve() / "Xcode.app" / "Contents"
            compiler = contents / "Developer" / "coremlcompiler"
            compiler.parent.mkdir(parents=True)
            compiler.write_text("compiler")
            metadata = contents / "version.plist"
            metadata.write_text("Xcode A")
            with (
                patch.object(build.subprocess, "run", return_value=SimpleNamespace(stdout=str(compiler))) as find,
                patch.object(build.platform, "version", return_value="host A") as host,
            ):
                selected, key = build.select_compiler("graph A")
                self.assertEqual(selected, compiler)
                self.assertNotEqual(key, build.select_compiler("graph B")[1])
                metadata.write_text("Xcode B")
                new_key = build.select_compiler("graph A")[1]
                self.assertNotEqual(key, new_key)
                host.return_value = "host B"
                self.assertNotEqual(new_key, build.select_compiler("graph A")[1])
                host.return_value = "host A"
                other = compiler.with_name("alternate-coremlcompiler")
                other.write_bytes(compiler.read_bytes())
                find.return_value = SimpleNamespace(stdout=str(other))
                self.assertNotEqual(new_key, build.select_compiler("graph A")[1])


if __name__ == "__main__":
    unittest.main()
