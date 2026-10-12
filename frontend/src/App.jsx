import { useEffect, useState } from "react"
import api from "./services/api"

const FEATURES = [
  { name: "duration_ms", label: "Duration (milliseconds)", help: "For example, 210000 is 3 minutes 30 seconds.", min: 1, max: 3600000, step: 1 },
  { name: "danceability", label: "Danceability", help: "Enter a value from 0 to 1.", min: 0, max: 1, step: "any" },
  { name: "energy", label: "Energy", help: "Enter a value from 0 to 1.", min: 0, max: 1, step: "any" },
  { name: "loudness", label: "Loudness (dB)", help: "Usually a negative number, such as -5.4.", min: -100, max: 20, step: "any" },
  { name: "speechiness", label: "Speechiness", help: "Enter a value from 0 to 1.", min: 0, max: 1, step: "any" },
  { name: "acousticness", label: "Acousticness", help: "Enter a value from 0 to 1.", min: 0, max: 1, step: "any" },
  { name: "instrumentalness", label: "Instrumentalness", help: "Enter a value from 0 to 1.", min: 0, max: 1, step: "any" },
  { name: "liveness", label: "Liveness", help: "Enter a value from 0 to 1.", min: 0, max: 1, step: "any" },
  { name: "valence", label: "Valence (musical positivity)", help: "Enter a value from 0 to 1.", min: 0, max: 1, step: "any" },
  { name: "tempo", label: "Tempo (BPM)", help: "Beats per minute, for example 120.", min: 0.01, max: 400, step: "any" },
]

const EMPTY_FEATURES = Object.fromEntries(FEATURES.map(({ name }) => [name, ""]))
const number = (value) => Number(value ?? 0).toLocaleString()
const fixed = (value, digits = 1) => Number(value).toFixed(digits)

export default function App() {
  const [data, setData] = useState(null)
  const [error, setError] = useState("")
  const [genres, setGenres] = useState([])
  const [genre, setGenre] = useState("")
  const [trackSearch, setTrackSearch] = useState("")
  const [tracks, setTracks] = useState([])
  const [tracksLoading, setTracksLoading] = useState(false)
  const [tracksError, setTracksError] = useState("")
  const [trackName, setTrackName] = useState("")
  const [artists, setArtists] = useState("")
  const [features, setFeatures] = useState(EMPTY_FEATURES)
  const [selectedTrack, setSelectedTrack] = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [predictionError, setPredictionError] = useState("")
  const [predicting, setPredicting] = useState(false)

  useEffect(() => {
    api.get("/api/all-results")
      .then((response) => setData(response.data))
      .catch(() => setError("Could not load the ML results. Check that the backend is running."))

    api.get("/api/genres")
      .then((response) => setGenres(response.data))
      .catch(() => setTracksError("Could not load the dataset genres."))
  }, [])

  useEffect(() => {
    let cancelled = false
    const timer = setTimeout(() => {
      setTracksLoading(true)
      setTracksError("")
      api.get("/api/tracks", { params: { query: trackSearch, genre, limit: 12 } })
        .then((response) => {
          if (!cancelled) setTracks(response.data.tracks)
        })
        .catch(() => {
          if (!cancelled) setTracksError("Could not load matching tracks. Check that the backend is running.")
        })
        .finally(() => {
          if (!cancelled) setTracksLoading(false)
        })
    }, 180)

    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [trackSearch, genre])

  const chooseTrack = (track) => {
    setTrackName(track.track_name)
    setArtists(track.artists)
    setFeatures(Object.fromEntries(FEATURES.map(({ name }) => [name, String(track.features[name]) ])))
    setSelectedTrack(track)
    setPrediction(null)
    setPredictionError("")
  }

  const updateFeature = (name, value) => {
    setFeatures((current) => ({ ...current, [name]: value }))
    setSelectedTrack(null)
    setPrediction(null)
  }

  const submitPrediction = async (event) => {
    event.preventDefault()
    setPredicting(true)
    setPrediction(null)
    setPredictionError("")

    const payload = {
      track_name: trackName,
      artists,
      ...Object.fromEntries(FEATURES.map(({ name }) => [name, Number(features[name])])),
    }

    try {
      const response = await api.post("/api/predict", payload)
      setPrediction(response.data)
    } catch (requestError) {
      setPredictionError(requestError.response?.data?.detail || "Could not calculate a result. Check that the backend is running.")
    } finally {
      setPredicting(false)
    }
  }

  return (
    <div className="app">
      <header className="hero">
        <div>
          <p className="eyebrow">SPOTIFY TRACK INTELLIGENCE</p>
          <h1>PlaylistPulse</h1>
          <p className="subtitle">Explore the songs in your dataset and try the trained models with track data.</p>
        </div>
        <div className="status"><span className="status-dot" />{error ? "API unavailable" : data ? "ML dashboard live" : "Loading results…"}</div>
      </header>

      <main>
        {error && <section className="error-box"><h2>Couldn’t load the dashboard</h2><p>{error}</p></section>}
        {!data && !error && <section className="loading">Loading dataset and model results…</section>}
        {data && <>
          <section className="overview">
            <div className="section-heading">
              <p className="eyebrow">DATASET OVERVIEW</p>
              <h2>Your Spotify catalogue, at a glance</h2>
              <p>Model scores use held-out data. Popularity is difficult to predict from audio alone, so compare several measures rather than relying on accuracy by itself.</p>
            </div>
            <div className="stats-grid">
              <article className="stat-card"><span>Tracks</span><strong>{number(data.dataset.rows)}</strong></article>
              <article className="stat-card"><span>Audio and metadata fields</span><strong>{number(data.dataset.columns)}</strong></article>
              <article className="stat-card"><span>Missing values</span><strong>{number(data.dataset.missing_values)}</strong></article>
              <article className="stat-card"><span>Popular tracks</span><strong>{number(data.dataset.class_distribution.popular)}</strong></article>
            </div>
          </section>

          <section className="overview explorer-section">
            <div className="section-heading">
              <p className="eyebrow">USE YOUR DATA</p>
              <h2>Pick a song or enter its audio features</h2>
              <p>Search the spreadsheet by song, artist, or genre. Choosing a row fills in its values; you can also type the values from Excel yourself.</p>
            </div>

            <div className="explorer-layout">
              <section className="explorer-panel track-browser">
                <div className="panel-heading">
                  <h3>Find a song in the dataset</h3>
                  <p>Tracks are ordered by the dataset’s popularity score.</p>
                </div>
                <label className="control-label" htmlFor="track-search">Song or artist</label>
                <input
                  id="track-search"
                  className="text-input"
                  type="search"
                  placeholder="Type a song or artist name"
                  value={trackSearch}
                  onChange={(event) => setTrackSearch(event.target.value)}
                />
                <label className="control-label" htmlFor="genre-filter">Genre</label>
                <select id="genre-filter" className="text-input" value={genre} onChange={(event) => setGenre(event.target.value)}>
                  <option value="">All genres</option>
                  {genres.map((item) => <option key={item} value={item}>{item}</option>)}
                </select>
                <div className="track-list" aria-live="polite">
                  {tracksLoading && <p className="list-message">Searching the dataset…</p>}
                  {!tracksLoading && tracksError && <p className="list-message error-text">{tracksError}</p>}
                  {!tracksLoading && !tracksError && tracks.length === 0 && <p className="list-message">No matching songs. Try a different name or genre.</p>}
                  {!tracksLoading && !tracksError && tracks.map((track) => (
                    <button
                      className={`track-row${selectedTrack?.track_id === track.track_id ? " selected" : ""}`}
                      key={`${track.track_id}-${track.track_genre}`}
                      type="button"
                      onClick={() => chooseTrack(track)}
                    >
                      <span className="track-details"><strong>{track.track_name}</strong><small>{track.artists} · {track.track_genre}</small></span>
                      <span className="track-score"><strong>{fixed(track.popularity, 0)}</strong><small>score</small></span>
                    </button>
                  ))}
                </div>
              </section>

              <section className="explorer-panel input-panel">
                <div className="panel-heading">
                  <h3>Enter track data</h3>
                  <p>Fill in the 10 audio-feature columns used by the prediction models.</p>
                </div>
                <form onSubmit={submitPrediction}>
                  <div className="identity-grid">
                    <label className="field"><span>Song name <small>optional</small></span><input className="text-input" value={trackName} onChange={(event) => { setTrackName(event.target.value); setSelectedTrack(null) }} placeholder="Song name" /></label>
                    <label className="field"><span>Artist <small>optional</small></span><input className="text-input" value={artists} onChange={(event) => { setArtists(event.target.value); setSelectedTrack(null) }} placeholder="Artist name" /></label>
                  </div>
                  <div className="feature-grid">
                    {FEATURES.map((field) => (
                      <label className="field" key={field.name}>
                        <span>{field.label}</span>
                        <input
                          className="text-input"
                          type="number"
                          required
                          min={field.min}
                          max={field.max}
                          step={field.step}
                          value={features[field.name]}
                          onChange={(event) => updateFeature(field.name, event.target.value)}
                        />
                        <small>{field.help}</small>
                      </label>
                    ))}
                  </div>
                  <button className="primary-button" type="submit" disabled={predicting}>
                    {predicting ? "Calculating…" : "Show track results"}
                  </button>
                </form>

                {predictionError && <p className="inline-error" role="alert">{predictionError}</p>}
                {prediction && <div className="prediction-result" aria-live="polite">
                  <div className="result-heading">
                    <div><p className="eyebrow">MODEL ESTIMATE</p><h4>{trackName || "Your track"}{artists ? ` · ${artists}` : ""}</h4></div>
                    {selectedTrack && <span className="actual-score">Dataset score: {fixed(selectedTrack.popularity, 0)}</span>}
                  </div>
                  <div className="prediction-grid">
                    {prediction.predicted_popularity !== undefined && <article className="prediction-card"><span>Predicted popularity</span><strong>{fixed(prediction.predicted_popularity)}</strong><small>Estimated score out of 100</small></article>}
                    {prediction.popular !== undefined && <article className="prediction-card"><span>Popular class</span><strong>{prediction.popular ? "Likely popular" : "Less likely popular"}</strong><small>{fixed((prediction.popular_probability ?? 0) * 100, 1)}% model probability</small></article>}
                    {prediction.cluster !== undefined && <article className="prediction-card"><span>Audio group</span><strong>Group {prediction.cluster + 1}</strong><small>Tracks with a similar audio profile</small></article>}
                  </div>
                  <p className="result-note">These are model estimates from audio features, not your personal Spotify play count.</p>
                </div>}
              </section>
            </div>
          </section>

          <section className="overview">
            <div className="section-heading"><p className="eyebrow">MODEL SNAPSHOT</p><h2>How the models performed</h2></div>
            <div className="model-grid">
              <article className="model-card"><div className="card-number">01 · REGRESSION</div><h3>Popularity score</h3><p>Estimate track popularity from audio features.</p><strong>R² {data.regression.metrics.r2 ?? "—"} · RMSE {data.regression.metrics.rmse ?? "—"}</strong></article>
              <article className="model-card"><div className="card-number">02 · CLASSIFICATION</div><h3>Popular or not</h3><p>Classify tracks using the project’s popularity threshold.</p><strong>F1 {data.classification.metrics?.f1 ?? "—"} · ROC AUC {data.classification.metrics?.roc_auc ?? "—"}</strong></article>
              <article className="model-card"><div className="card-number">03 · CLUSTERING</div><h3>Audio groups</h3><p>Group tracks with similar audio profiles.</p><strong>{data.clustering.clusters ?? "—"} clusters · MiniBatch K-Means</strong></article>
            </div>
          </section>

          <section className="overview">
            <div className="section-heading"><p className="eyebrow">POPULAR GENRES</p><h2>Most represented genres</h2></div>
            <div className="model-grid">
              {data.eda.top_genres.slice(0, 6).map(({ genre: genreName, count }, index) => <article className="model-card genre-card" key={genreName}><div className="card-number">{String(index + 1).padStart(2, "0")}</div><h3>{genreName}</h3><strong>{number(count)} tracks</strong></article>)}
            </div>
          </section>
          <p className="footnote">The dataset popularity score is not a personal listening count. To find songs you personally play most, the app would need your Spotify streaming-history data. Monitoring compares two halves of this dataset; it is not a live production monitor.</p>
        </>}
      </main>
    </div>
  )
}
