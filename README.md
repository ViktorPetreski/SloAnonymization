# Slovenian text anonymization
### Master's thesis of Viktor Petreski 

## Requrements
- docker
- docker-compose

## Running the pipeline

Each image should be first built:
```bash
cd src
docker build --tag pipeline -f Dockerfile-pipeline .  --progress=plain
docker build --tag ner -f Dockerfile-ner .  --progress=plain
docker build --tag pos -f Dockerfile-pos .  --progress=plain
docker build --tag coref -f Dockerfile-coref .  --progress=plain
docker build --tag frontend -f Dockerfile-frontend .  --progress=plain 
```

Then, running the command `docker-compose -f docker-compose.yml up`  would start the containers after 3-4 minutes. 

The images in the `docker-compose.yml` file can be changed to the pre-compiled ones found on my [DockerHub](https://hub.docker.com/u/petreskiv).

The web tool is listening on `http://localhost:8080`

The pipeline API is listening on `http://localhost:5050/`.

The endpoint __anonymize__ accepts *POST* request with body: 
```json
{
    "text": "Slovenske prvake že v nedeljo, 23. septembra, v velenjski Rdeči dvorani čaka večni derbi z Gorenjem v okviru tretjega kroga lige NLB, v četrtek, 29. septembra, pa še gostovanje pri španski Barceloni. Katalonski velikan, kjer igrata tudi slovenska reprezentanta Blaž Janc in Domen Makuc, je v zadnjih dveh sezonah slavil v ligi prvakov.",
    "mode": ""
}
```
The _mode_ parameter accepts:
- low
- medium
- high
- {empty string}
