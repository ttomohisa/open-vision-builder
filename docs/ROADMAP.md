# Roadmap

| Phase | PR scope | Exit condition |
| --- | --- | --- |
| P0 | Product docs, agent rules, model candidates, verification gates | Requirements and unknowns recorded |
| P1 | Feasibility spike: MobileNetV4 Conv Small and D-FINE Nano | Real browser weight updates, complete ONNX export, Python inference parity, measured memory/accuracy |
| P2 | Project import, local storage, annotation UI and backups | Images, classes and bounding boxes survive export/import |
| P3 | Training/evaluation workflows | Re-training and held-out evaluation with failure review |
| P4 | Python/API/static inference deployment kits | Standalone artifacts work outside builder |
| P5 | Browser compatibility, edge profiles, docs, release checks | All agreed gates evidenced |

ONNX is **P1**, not deferred until deployment-kit work. Make the first runnable spike available through a verified GitHub Pages preview; PR preview workflow must be implemented explicitly, not assumed built-in.
