curl -X POST "http://127.0.0.1:8000/run" \
  -F "dataset=@ml-100k.zip" \
  -F "backend=mock" \
  -F "max_rows=500" \
  -F 'top_k_json=[5,10,20]'
