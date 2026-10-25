import { useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '.'
import type { ApiError, FeedFilters, InsightCard, JobView, Ready, Role, ScenarioSet } from './types'

export const useCompanies = (login?: string) =>
  useQuery({ queryKey: ['companies', login], queryFn: () => api.companies(), enabled: !!login })

export const useCompany = (id: string) =>
  useQuery({ queryKey: ['company', id], queryFn: () => api.company(id), enabled: !!id })

export const useFeed = (companyId: string, filters: FeedFilters = {}) =>
  useQuery({ queryKey: ['feed', companyId, filters], queryFn: () => api.feed(companyId, filters), enabled: !!companyId, placeholderData: prev => prev })

export const useSummary = (companyId: string) =>
  useQuery({ queryKey: ['summary', companyId], queryFn: () => api.summary(companyId), enabled: !!companyId })

// Готовый объект приходит сразу (200), новый — через задачу (202) с опросом poll_url.
function useGenerated<T>(key: unknown[], start: () => Promise<Ready<T>>, pick: (j: JobView) => T | null, enabled: boolean, interval: number) {
  const qc = useQueryClient()
  const first = useQuery({ queryKey: key, queryFn: start, enabled, staleTime: Infinity, retry: false })
  const jobId = first.data?.status === 'pending' ? first.data.job.job_id : undefined
  const job = useQuery({
    queryKey: ['job', jobId],
    queryFn: () => api.job(jobId!),
    enabled: !!jobId,
    staleTime: Infinity,
    retry: false,
    refetchInterval: q => (q.state.data && ['done', 'failed'].includes(q.state.data.status) ? false : interval),
  })
  const jobView = job.data ?? (first.data?.status === 'pending' ? first.data.job : undefined)
  const data = first.data?.status === 'ready' ? first.data.data : job.data?.status === 'done' ? pick(job.data) ?? undefined : undefined
  const failed = job.data?.status === 'failed' && job.data.error ? { status: 502, ...job.data.error } as ApiError : undefined
  return {
    data,
    job: jobView,
    pending: enabled && !data && !failed && !first.error && (first.isLoading || !!jobId),
    error: (first.error as ApiError | null) ?? failed ?? (job.error as ApiError | null) ?? undefined,
    retry() { qc.removeQueries({ queryKey: ['job', jobId] }); qc.invalidateQueries({ queryKey: key }) },
  }
}

export function useCard(companyId: string, itemId: string, role?: Role) {
  const r = useGenerated<InsightCard>(['card', companyId, itemId, role ?? 'own'], () => api.card(companyId, itemId, role), j => j.card, !!companyId && !!itemId, 1000)
  return { card: r.data, job: r.job, pending: r.pending, error: r.error, retry: r.retry }
}

// Сценарии строятся по запросу: около рубля и десятков секунд, поэтому не автоматически.
export function useScenarios(companyId: string, itemId: string, role: Role | undefined, enabled: boolean) {
  const r = useGenerated<ScenarioSet>(['scenarios', companyId, itemId, role ?? 'own'], () => api.scenarios(companyId, itemId, role), j => j.scenarios, enabled, 2000)
  return { set: r.data, job: r.job, pending: r.pending, error: r.error, retry: r.retry }
}
