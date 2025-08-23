# Streamlit Football App (RapidAPI: API-Football)

Valmis mobiiliystävällinen Streamlit-sovellus: selaa otteluita, hae kertoimet ja tee **EV/Kelly**-laskenta, sekä listaa **value-bets**.

## Pika-ajo (paikallisesti)
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

> Avaimena käytetään `.streamlit/secrets.toml` tiedostoa, joka on tässä paketissa valmiiksi asetettu **vain sinulle**.

## Streamlit Cloud / GitHub
1) **ÄLÄ** puske omaa avaintasi julkiseen repoosi. Poista `.streamlit/secrets.toml` ennen pushia.  
2) Aseta **Secrets** Streamlit Cloudissa:
```
RAPIDAPI_KEY = "91e39142e1c544758e7b3d9f1edeaf1c"
RAPIDAPI_HOST = "api-football-v1.p.rapidapi.com"
```
3) Aja appi Cloudissa.

## Toiminnot
- **Fixtures**: päivämäärän/maan/ligan mukaan (esiasetuksissa isoimmat liigalistat).
- **Odds**: hae fikstuurin kertoimet, laske implied, EV ja Kelly.
- **Value Bets**: käy läpi valitun liigan tulevat matsit, suodattaa kohteet minimi-EV:llä ja ehdottaa Kelly-panososuuden.

