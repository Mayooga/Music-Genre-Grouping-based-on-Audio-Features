import { useEffect, useMemo, useState } from 'react'
import Papa from 'papaparse'
import type { ParseResult } from 'papaparse'
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { ClusterReport, ClusterSummaryRow, TrackRow } from './types'

const DATA_BASE = '/data'

async function loadJson<T>(path: string): Promise<T> {
  const res = await fetch(path)
  if (!res.ok) {
    throw new Error(`Failed to load ${path}: ${res.status}`)
  }
  return res.json() as Promise<T>
}

async function loadCsv<T>(path: string): Promise<T[]> {
  return new Promise((resolve, reject) => {
    Papa.parse<T>(path, {
      download: true,
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: (result: ParseResult<T>) => resolve(result.data as T[]),
      error: reject,
    })
  })
}

type LoadingState = 'idle' | 'loading' | 'ready' | 'error'

const featureColumnsToShow = ['tempo', 'energy', 'danceability', 'valence', 'acousticness']
const trackColumns = ['track_name', 'artist_name', 'tempo', 'energy', 'danceability', 'valence', 'popularity']

const clusterExplanations = [
  {
    id: 0,
    title: 'Cluster 0 – Energetic EDM / Pop',
    bullets: ['High tempo', 'High energy', 'Loudness high'],
    summary: 'Very energetic dance / pop songs that feel intense and upbeat.',
  },
  {
    id: 1,
    title: 'Cluster 1 – Acoustic & Soft',
    bullets: ['Acousticness high', 'Speechiness low'],
    summary: 'Soft, acoustic music such as calm pop, ballads or light background tracks.',
  },
  {
    id: 2,
    title: 'Cluster 2 – Rap / Hip-Hop',
    bullets: ['Speechiness high', 'Energy medium to high'],
    summary: 'Tracks with lots of words and clear vocals, similar to Hip‑Hop or Rap.',
  },
  {
    id: 3,
    title: 'Cluster 3 – Instrumental / Ambient',
    bullets: ['Instrumentalness high', 'Speechiness very low'],
    summary: 'Mostly instrumental songs or ambient tracks with very few vocals.',
  },
  {
    id: 4,
    title: 'Cluster 4 – Mainstream Pop',
    bullets: ['Medium tempo', 'Balanced energy', 'Danceability medium'],
    summary: 'Typical chart‑style pop songs: not too loud, not too quiet, easy to listen to.',
  },
  {
    id: 5,
    title: 'Cluster 5 – Chill & Relaxed',
    bullets: ['Lower tempo', 'Lower energy', 'Valence medium'],
    summary: 'Laid‑back songs that are good for relaxing, studying or late‑night playlists.',
  },
  {
    id: 6,
    title: 'Cluster 6 – Danceable Pop / R&B',
    bullets: ['Danceability high', 'Energy medium', 'Tempo medium'],
    summary: 'Groovy tracks that are quite danceable but not as aggressive as EDM.',
  },
  {
    id: 7,
    title: 'Cluster 7 – Rare / Niche',
    bullets: ['Very small cluster size', 'Unusual feature mix'],
    summary: 'A small group of special songs that do not fit clearly into the other clusters.',
  },
]

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric-card">
      <p className="metric-label">{label}</p>
      <p className="metric-value">{value}</p>
    </div>
  )
}

function formatNumber(value?: number | null, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '—'
  }
  return Number(value).toFixed(digits)
}

export default function App() {
  const [state, setState] = useState<LoadingState>('idle')
  const [error, setError] = useState<string | null>(null)
  const [report, setReport] = useState<ClusterReport | null>(null)
  const [summary, setSummary] = useState<ClusterSummaryRow[]>([])
  const [tracks, setTracks] = useState<TrackRow[]>([])
  const [search, setSearch] = useState('')
  const [selectedCluster, setSelectedCluster] = useState<string>('all')

  useEffect(() => {
    const run = async () => {
      try {
        setState('loading')
        const [reportData, summaryData, trackData] = await Promise.all([
          loadJson<ClusterReport>(`${DATA_BASE}/cluster_report.json`),
          loadCsv<ClusterSummaryRow>(`${DATA_BASE}/cluster_summary.csv`),
          loadCsv<TrackRow>(`${DATA_BASE}/clustered_audio_features_sample.csv`),
        ])
        setReport(reportData)
        setSummary(summaryData.filter((row) => row.cluster_label !== undefined))
        setTracks(trackData.filter((row) => row.cluster_label !== undefined))
        setState('ready')
      } catch (err) {
        console.error(err)
        setError(err instanceof Error ? err.message : 'Failed to load data')
        setState('error')
      }
    }
    run()
  }, [])

  const clusterSizes = useMemo(() => {
    if (!report) return []
    return Object.entries(report.cluster_sizes ?? {}).map(([name, value]) => ({
      name,
      value,
    }))
  }, [report])

  const filteredTracks = useMemo(() => {
    if (!tracks.length) return []
    return tracks
      .filter((track) => {
        if (selectedCluster !== 'all' && `${track.cluster_label}` !== selectedCluster) {
          return false
        }
        if (!search.trim()) return true
        const haystack = `${track.track_name ?? ''} ${track.artist_name ?? ''}`.toLowerCase()
        return haystack.includes(search.trim().toLowerCase())
      })
      .slice(0, 200)
  }, [tracks, search, selectedCluster])

  const clusterOptions = useMemo(() => {
    const unique = new Set(summary.map((row) => `${row.cluster_label}`))
    return Array.from(unique).sort()
  }, [summary])

  const featureColumns = useMemo(() => {
    if (!summary.length) return featureColumnsToShow
    const first = summary[0]
    const cols = Object.keys(first)
      .filter((key) => key !== 'cluster_label' && key !== 'count')
      .slice(0, 6)
    return cols.length ? cols : featureColumnsToShow
  }, [summary])

  const statusMessage = {
    loading: 'Loading cluster data…',
    error: error ?? 'Unable to load data.',
    idle: '',
    ready: '',
  }[state]

  return (
    <main className="page">
      <header className="hero">
        <div>
          <h1>Music Genre Grouping Dashboard</h1>
        </div>
      </header>

      {state !== 'ready' && (
        <section className="status">
          <p>{statusMessage}</p>
        </section>
      )}

      {state === 'ready' && report && (
        <>
          <section className="grid">
            <MetricCard label="Tracks clustered" value={report.n_samples.toLocaleString()} />
            <MetricCard label="Number of clusters" value={report.n_clusters.toString()} />
          </section>

          <section className="panel">
            <div className="panel-header">
              <div>
                <h2>Cluster Distribution</h2>
                <p>Number of tracks that fall into each discovered cluster.</p>
              </div>
            </div>
            <div className="chart-wrapper">
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={clusterSizes}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="value" radius={[4, 4, 0, 0]} fill="#6366f1" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </section>

          <section className="panel">
            <div className="panel-header">
              <div>
                <h2>What Each Cluster Means</h2>
                <p>Short descriptions so you can quickly understand the type of songs in each group.</p>
              </div>
            </div>
            <div className="cluster-explanations">
              {clusterExplanations.map((cluster) => (
                <div key={cluster.id} className="cluster-card">
                  <h3>{cluster.title}</h3>
                  <ul>
                    {cluster.bullets.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                  <p className="hint">{cluster.summary}</p>
                </div>
              ))}
            </div>
          </section>

          <section className="panel">
            <div className="panel-header">
              <div>
                <h2>Cluster Feature Signatures</h2>
                <p>Average audio feature values per cluster.</p>
              </div>
            </div>
            <div className="table-wrapper">
              <table>
                <thead>
                  <tr>
                    <th>Cluster</th>
                    {featureColumns.map((col) => (
                      <th key={col}>{col}</th>
                    ))}
                    <th>Tracks</th>
                  </tr>
                </thead>
                <tbody>
                  {summary.map((row) => (
                    <tr key={row.cluster_label}>
                      <td>#{row.cluster_label}</td>
                      {featureColumns.map((col) => (
                        <td key={col}>{formatNumber(Number(row[col]))}</td>
                      ))}
                      <td>{row.count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section className="panel">
            <div className="panel-header">
              <div>
                <h2>Track Explorer</h2>
                <p>Sampled tracks with their assigned cluster label.</p>
              </div>
              <div className="filters">
                <input
                  type="text"
                  placeholder="Search by track or artist"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                />
                <select value={selectedCluster} onChange={(e) => setSelectedCluster(e.target.value)}>
                  <option value="all">All clusters</option>
                  {clusterOptions.map((option) => (
                    <option key={option} value={option}>
                      Cluster {option}
                    </option>
                  ))}
                </select>
              </div>
            </div>
            <div className="table-wrapper">
              <table>
                <thead>
                  <tr>
                    {trackColumns.map((col) => (
                      <th key={col}>{col.replace('_', ' ')}</th>
                    ))}
                    <th>Cluster</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredTracks.map((track, idx) => (
                    <tr key={`${track.track_id}-${idx}`}>
                      {trackColumns.map((col) => (
                        <td key={col}>
                          {typeof track[col] === 'number' ? formatNumber(Number(track[col])) : (track[col] as string) ?? '—'}
                        </td>
                      ))}
                      <td>
                        <span className="cluster-pill">#{track.cluster_label}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="hint">Showing up to 200 sampled tracks for quick browsing.</p>
          </section>
        </>
      )}

      {state === 'error' && (
        <section className="status error">
          <p>{error}</p>
        </section>
      )}
    </main>
  )
}

