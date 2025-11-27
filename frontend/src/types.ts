export type ClusterReport = {
  n_samples: number
  n_clusters: number
  silhouette_score?: number
  inertia?: number
  avg_log_likelihood?: number
  cluster_sizes: Record<string, number>
}

export type ClusterSummaryRow = {
  cluster_label: string | number
  count: number
} & Record<string, number>

export type TrackRow = {
  track_id?: string
  track_name?: string
  artist_name?: string
  cluster_label: string | number
} & Record<string, string | number | undefined>


