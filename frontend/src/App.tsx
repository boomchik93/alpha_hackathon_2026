import type { ReactNode } from 'react'
import { BrowserRouter, HashRouter, Navigate, Route, Routes } from 'react-router-dom'
import { useSession } from './auth/session'
import { Shell } from './components/Shell'
import { FinancesPage, TasksPage } from './pages/BankPages'
import FeedPage from './pages/FeedPage'
import HistoryPage from './pages/HistoryPage'
import HomePage from './pages/HomePage'
import InsightPage from './pages/insight/InsightPage'
import { SourcePage, TracePage } from './pages/insight/SourceTrace'
import LoginPage from './pages/LoginPage'
import MonitoringPage from './pages/MonitoringPage'
import ProfilePage from './pages/ProfilePage'

// Демо-сборка (один HTML-файл) работает без сервера, поэтому там адреса через #.
const Router = import.meta.env.MODE === 'demo' ? HashRouter : BrowserRouter

function Private({ children }: { children: ReactNode }) {
  const { user, ready } = useSession()
  // Пока проверяем сохранённый токен (GET /me), не уводим на вход: иначе обновление страницы выбрасывает из сессии
  if (!ready) return <div className="boot-screen" aria-busy="true"><span className="spinner red" /></div>
  return user ? <Shell>{children}</Shell> : <Navigate to="/login" replace />
}

export default function App() {
  return <Router basename={import.meta.env.BASE_URL}>
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route path="/" element={<Private><HomePage /></Private>} />
      <Route path="/feed" element={<Private><FeedPage /></Private>} />
      <Route path="/insight/:id" element={<Private><InsightPage /></Private>} />
      <Route path="/source/:itemId/:sourceId" element={<Private><SourcePage /></Private>} />
      <Route path="/trace/:itemId" element={<Private><TracePage /></Private>} />
      <Route path="/profile" element={<Private><ProfilePage /></Private>} />
      <Route path="/history" element={<Private><HistoryPage /></Private>} />
      <Route path="/monitoring" element={<Private><MonitoringPage /></Private>} />
      <Route path="/finances" element={<Private><FinancesPage /></Private>} />
      <Route path="/tasks" element={<Private><TasksPage /></Private>} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  </Router>
}
