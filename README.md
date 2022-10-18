# Slovenian text anonymization
### Master's thesis of Viktor Petreski
---
## Running the pipeline

```bash
cd src
docker-compose -f docker-compose.yml up
``

The web tool is listening on `http://localhost:8080`

The pipeline API is listening on `http://localhost:5050/`.

The endpoint _anonymize_ accepts *POST* request with body: 
```json
{
    "text": "Slovenske prvake že v nedeljo, 23. septembra, v velenjski Rdeči dvorani čaka večni derbi z Gorenjem v okviru tretjega kroga lige NLB, v četrtek, 29. septembra, pa še gostovanje pri španski Barceloni. Katalonski velikan, kjer igrata tudi slovenska reprezentanta Blaž Janc in Domen Makuc, je v zadnjih dveh sezonah slavil v ligi prvakov.",
    "mode": ""
}
```

