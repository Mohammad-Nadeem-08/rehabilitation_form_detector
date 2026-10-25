import React from 'react';
import { Activity, Camera, History, Cpu, FileText, CheckCircle2, AlertTriangle } from 'lucide-react';

interface NavbarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  backendHealthy: boolean;
  modelLoaded: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  backendHealthy,
  modelLoaded,
}) => {
  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: Activity },
    { id: 'reference', label: 'Exercise Reference', icon: FileText },
    { id: 'assessment', label: 'Live Assessment', icon: Camera },
    { id: 'model-lab', label: 'Model Lab (LSTM)', icon: Cpu },
    { id: 'history', label: 'Session History', icon: History },
  ];

  return (
    <header style={{
      borderBottom: '1px solid var(--border-subtle)',
      background: 'rgba(10, 17, 40, 0.95)',
      backdropFilter: 'blur(10px)',
      position: 'sticky',
      top: 0,
      zIndex: 50,
      padding: '0 20px',
    }}>
      <div style={{
        maxWidth: '1360px',
        margin: '0 auto',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        height: '70px',
      }}>
        {/* Brand logo & title */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', cursor: 'pointer' }} onClick={() => setActiveTab('dashboard')}>
          <div style={{
            width: '40px',
            height: '40px',
            borderRadius: '10px',
            background: 'linear-gradient(135deg, #00E5FF 0%, #0077B6 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 0 15px rgba(0, 229, 255, 0.4)',
          }}>
            <Activity size={24} color="#070B19" strokeWidth={2.5} />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontSize: '1.25rem', fontWeight: 800, letterSpacing: '-0.02em', color: '#F8FAFC' }}>
                RehabVision <span style={{ color: '#00E5FF' }}>AI</span>
              </span>
              <span className="badge badge-teal" style={{ fontSize: '0.65rem', padding: '2px 6px' }}>
                v1.0 MVP
              </span>
            </div>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)' }}>
              Biomechanics & LSTM Form Detection
            </div>
          </div>
        </div>

        {/* Navigation tabs */}
        <nav style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  padding: '8px 16px',
                  borderRadius: '8px',
                  border: isActive ? '1px solid var(--border-active)' : '1px solid transparent',
                  background: isActive ? 'rgba(0, 229, 255, 0.1)' : 'transparent',
                  color: isActive ? '#00E5FF' : 'var(--text-secondary)',
                  fontWeight: isActive ? 600 : 500,
                  fontSize: '0.875rem',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease',
                }}
              >
                <Icon size={16} />
                {item.label}
              </button>
            );
          })}
        </nav>

        {/* System telemetry badges */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div
            title={backendHealthy ? 'FastAPI Backend Online' : 'Backend Connecting...'}
            className={backendHealthy ? 'badge badge-green' : 'badge badge-amber'}
            style={{ fontSize: '0.7rem' }}
          >
            {backendHealthy ? <CheckCircle2 size={12} /> : <AlertTriangle size={12} />}
            API {backendHealthy ? 'Online' : 'Offline'}
          </div>

          <div
            title={modelLoaded ? 'PyTorch LSTM (64x80) Loaded' : 'LSTM Checkpoint Unavailable'}
            className={modelLoaded ? 'badge badge-teal' : 'badge badge-amber'}
            style={{ fontSize: '0.7rem' }}
          >
            <Cpu size={12} />
            LSTM {modelLoaded ? 'Ready' : 'Checking'}
          </div>
        </div>
      </div>
    </header>
  );
};
