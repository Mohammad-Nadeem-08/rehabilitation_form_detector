import React, { useEffect, useState } from 'react';
import {
  Activity, Camera, FileText, Cpu, ArrowRight, Download,
  CheckCircle, AlertCircle, TrendingUp
} from 'lucide-react';
import { MetricCard } from '../components/MetricCard';
import { fetchProgress, getReportDownloadUrl } from '../services/api';
import { ProgressSummary, SessionData } from '../types';
import {
  ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, CartesianGrid
} from 'recharts';

interface DashboardPageProps {
  setActiveTab: (tab: string) => void;
  onSelectSession: (session: SessionData) => void;
}

export const DashboardPage: React.FC<DashboardPageProps> = ({
  setActiveTab,
  onSelectSession,
}) => {
  const [progress, setProgress] = useState<ProgressSummary | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      const data = await fetchProgress();
      setProgress(data);
    } catch (err) {
      console.error('Failed to load dashboard data:', err);
    } finally {
      setLoading(false);
    }
  };

  // Prepare chart data from recent sessions
  const chartData = (progress?.recent_sessions || [])
    .slice()
    .reverse()
    .map((s, idx) => ({
      name: `Sess ${idx + 1}`,
      score: s.comparison_score || 0,
      reps: s.repetitions_completed || 0,
      date: new Date(s.timestamp).toLocaleDateString(),
    }));

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Hero Banner with Quick Actions */}
      <div className="glass-panel" style={{
        padding: '28px',
        background: 'linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(10, 17, 40, 0.95) 100%)',
        border: '1px solid var(--border-active)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '20px' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
              <span className="badge badge-teal">Project Scope: Phase 1 — Phase 11</span>
              <span className="badge badge-green">Target: Standing Right-Arm Shoulder Abduction</span>
            </div>
            <h1 style={{ fontSize: '1.85rem', fontWeight: 800, color: '#F8FAFC', marginBottom: '8px' }}>
              Rehabilitation Kinematics & Deep Learning Form Hub
            </h1>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', maxWidth: '640px' }}>
              Real-time MediaPipe pose estimation, dynamic time warping reference comparison, and genuine PyTorch LSTM inference.
            </p>
          </div>

          <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
            <button
              onClick={() => setActiveTab('assessment')}
              className="btn btn-primary"
              style={{ padding: '12px 20px' }}
            >
              <Camera size={18} />
              Start Live Assessment
            </button>
            <button
              onClick={() => setActiveTab('reference')}
              className="btn btn-secondary"
              style={{ padding: '12px 20px' }}
            >
              <FileText size={18} />
              Record Reference
            </button>
          </div>
        </div>
      </div>

      {/* Primary KPI Metrics Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '16px' }}>
        <MetricCard
          title="Gold References"
          value={progress?.total_references ?? 0}
          subtitle="Saved exercise standards"
          icon={FileText}
          badge={{ text: 'Baseline Active', variant: 'teal' }}
        />

        <MetricCard
          title="Sessions Completed"
          value={progress?.total_sessions ?? 0}
          subtitle="Total recorded workouts"
          icon={Activity}
          badge={{ text: 'Local SQLite', variant: 'green' }}
        />

        <MetricCard
          title="Total Repetitions"
          value={progress?.total_repetitions ?? 0}
          subtitle="Shoulder abduction reps"
          icon={TrendingUp}
          badge={{ text: 'FSM Validated', variant: 'teal' }}
        />

        <MetricCard
          title="Avg Comparison Score"
          value={progress?.average_comparison_score ? `${progress.average_comparison_score}%` : 'N/A'}
          subtitle="DTW Movement Similarity"
          icon={CheckCircle}
          badge={{ text: 'Prototype Index', variant: 'green' }}
        />
      </div>

      {/* Historical Movement Quality Curve */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '18px' }}>
          <div>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#F8FAFC' }}>
              Historical Movement Quality Trend
            </h3>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              Evolution of Reference Similarity Index (%) across recorded repetitions and sessions.
            </p>
          </div>
          <span className="badge badge-teal">DTW Trajectory Variance</span>
        </div>

        <div style={{ width: '100%', height: 260 }}>
          {chartData.length === 0 ? (
            <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
              No session trends available yet. Complete a live assessment to generate history.
            </div>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData} margin={{ top: 10, right: 15, left: -15, bottom: 0 }}>
                <defs>
                  <linearGradient id="scoreGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#00E5FF" stopOpacity={0.4}/>
                    <stop offset="95%" stopColor="#00E5FF" stopOpacity={0.0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(51, 65, 85, 0.4)" />
                <XAxis dataKey="name" stroke="#64748B" fontSize={11} />
                <YAxis domain={[0, 100]} stroke="#64748B" fontSize={11} ticks={[0, 25, 50, 75, 100]} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0F172A',
                    borderColor: 'rgba(0, 229, 255, 0.3)',
                    borderRadius: '8px',
                    fontSize: '0.8rem',
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="score"
                  name="Similarity Index (%)"
                  stroke="#00E5FF"
                  strokeWidth={2.5}
                  fillOpacity={1}
                  fill="url(#scoreGradient)"
                />
              </AreaChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      {/* Recent Sessions Table */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
          <div>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#F8FAFC' }}>
              Recent Assessment Sessions
            </h3>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              Inspect recorded patient movement, biomechanical deviations, and clinical PDF reports.
            </p>
          </div>
          <button onClick={() => setActiveTab('history')} className="btn btn-secondary" style={{ fontSize: '0.8rem', padding: '6px 12px' }}>
            View All <ArrowRight size={14} />
          </button>
        </div>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-secondary)', textAlign: 'left' }}>
                <th style={{ padding: '10px 12px' }}>Session ID</th>
                <th style={{ padding: '10px 12px' }}>Timestamp</th>
                <th style={{ padding: '10px 12px' }}>Reps</th>
                <th style={{ padding: '10px 12px' }}>Duration</th>
                <th style={{ padding: '10px 12px' }}>Peak ROM</th>
                <th style={{ padding: '10px 12px' }}>DTW Score</th>
                <th style={{ padding: '10px 12px' }}>LSTM Status</th>
                <th style={{ padding: '10px 12px', textAlign: 'right' }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {(progress?.recent_sessions || []).slice(0, 5).map((session) => (
                <tr
                  key={session.id}
                  style={{ borderBottom: '1px solid rgba(51, 65, 85, 0.3)', transition: 'background 0.15s ease' }}
                  onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = 'rgba(30, 41, 59, 0.4)')}
                  onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
                >
                  <td style={{ padding: '12px', fontFamily: 'var(--font-mono)', color: '#00E5FF' }}>
                    {session.id}
                    {session.is_demo ? <span className="badge badge-amber" style={{ fontSize: '0.6rem', marginLeft: '6px' }}>DEMO</span> : null}
                  </td>
                  <td style={{ padding: '12px', color: 'var(--text-secondary)' }}>
                    {new Date(session.timestamp).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}
                  </td>
                  <td style={{ padding: '12px', fontWeight: 600 }}>{session.repetitions_completed}</td>
                  <td style={{ padding: '12px', color: 'var(--text-secondary)' }}>{session.duration_seconds.toFixed(1)}s</td>
                  <td style={{ padding: '12px' }}>{session.angle_stats?.peak_elevation?.toFixed(1) || 'N/A'}°</td>
                  <td style={{ padding: '12px' }}>
                    <span className={`badge ${(session.comparison_score ?? 0) >= 80 ? 'badge-green' : 'badge-amber'}`}>
                      {session.comparison_score ? `${session.comparison_score.toFixed(1)}%` : 'N/A'}
                    </span>
                  </td>
                  <td style={{ padding: '12px' }}>
                    {session.lstm_prediction ? (
                      <span className="badge badge-teal">Class {session.lstm_prediction.predicted_class}</span>
                    ) : (
                      <span className="badge badge-amber" title="Live LSTM requires verified mocap adapter">Requires Validation</span>
                    )}
                  </td>
                  <td style={{ padding: '12px', textAlign: 'right' }}>
                    <div style={{ display: 'flex', gap: '8px', justifyContent: 'flex-end' }}>
                      <button
                        onClick={() => onSelectSession(session)}
                        className="btn btn-secondary"
                        style={{ padding: '6px 10px', fontSize: '0.75rem' }}
                      >
                        Details
                      </button>
                      <a
                        href={getReportDownloadUrl(session.id)}
                        target="_blank"
                        rel="noreferrer"
                        className="btn btn-primary"
                        style={{ padding: '6px 10px', fontSize: '0.75rem', textDecoration: 'none' }}
                      >
                        <Download size={12} />
                        PDF
                      </a>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
