# Bayesian Optimization for Game Engines

Bayesian Optimization for Game Engines provides a shared Python optimizer and native Unity and Unreal integrations. The backends support single objective optimization, multi objective optimization, CABOP, DynamicBO, contextual optimization, and MetaTAF. Both engine packages use the same configuration and observation protocol.

The algorithms are maintained in `Backend/`. Unity and Unreal runtime copies are generated from that source. Each release contains independently installable engine packages built from one repository revision. This arrangement keeps numerical fixes and experiment validation consistent across engines.

| Directory | Purpose |
| --- | --- |
| `Backend/` | Canonical Python implementations, requirements, licenses, and example data |
| `Unity/` | Engine wrapper, analytic example, tests, and rendering tools |
| `Unreal/` | C++ plugin, Blueprint component, analytic actor, and Python launcher |
| `tests/` | Shared Python tests and distribution checks |
| `tools/` | Backend synchronization and release packaging |
| `docs/` | Architecture, validation evidence, and publication status |

Download an engine package from [GitHub Releases](https://github.com/M-Colley/Bayesian-Optimization-for-Game-Engines/releases). Import the Unity `.unitypackage`, or copy the Unreal plugin into a C++ Unreal project. The Unity ZIP provides the complete example project. The Unreal ZIP includes Python source and launch instructions. See the [Unity guide](Unity/README.md) and [Unreal guide](Unreal/README.md).

Both integrations require Python 3.13 and the pinned dependencies in `Backend/requirements.txt`. Configure a writable log directory and retain the study configuration, environment, seed, and source revision with each experiment. The connection uses loopback TCP. Different ports allow concurrent studies.

The Unity core runs without QuestionnaireToolkit. Custom feedback interfaces can use `BayesianOptimization.Integration` contracts or submit telemetry directly. QuestionnaireToolkit and its native file browsers and sample assets are excluded from these packages. Unity rendering tools generate materials for Built-in, URP, and HDRP through public Editor APIs. They preserve the project's pipeline selection and installed packages.

Numerical changes belong in `Backend/`. Regenerate and verify both runtime copies before committing:

```text
python tools/sync_backends.py
python tools/sync_backends.py --check
python -m pytest -q
```

Build release packages with the installed Unity Editor:

```text
python tools/build_packages.py --version 2.0.0-rc.1 --output ReleaseArtifacts --unity-editor <Unity executable>
```

The packager validates runtime copies, creates both engine ZIP files, exports the Unity asset package, and writes SHA256 checksums. It rejects bundled installers and commercial questionnaire assets. The [validation record](docs/validation.md) identifies tested versions. The [publication assessment](docs/publication-status.md) lists the remaining Asset Store work.

This project derives from [Bayesian Optimization for Unity](https://github.com/Pascal-Jansen/Bayesian-Optimization-for-Unity) by Pascal Jansen and Mark Colley. The original MIT notice is retained. CABOP and DynamicBO retain their component licenses and provenance. See [LICENSE](LICENSE) and [Third Party Notices](THIRD-PARTY-NOTICES.txt). Refactoring and integration used OpenAI Codex assistance.
