`model.onnx.b64` is the exact byte-for-byte base64 payload of the promoted ONNX model. Restore with:

```bash
base64 -d model.onnx.b64 > model.onnx
sha256sum -c <(printf '126beb05365eaf1edaefb6797406607515259cfc4a0eff2b52e6ac96987ae85c  model.onnx\n')
```
