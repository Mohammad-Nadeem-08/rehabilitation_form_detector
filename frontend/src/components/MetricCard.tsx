import React from 'react';
import { LucideIcon } from 'lucide-react';

interface MetricCardProps {
  title: string;
  value: string | number;
  unit?: string;
  subtitle?: string;
  icon: LucideIcon;
  badge?: {
    text: string;
    variant: 'teal' | 'green' | 'amber' | 'red';
  };
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  unit,
  subtitle,
  icon: Icon,
  badge,
}) => {
  return (
    <div className="glass-panel" style={{ padding: '20px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
        <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.04em', fontWeight: 600 }}>
          {title}
        </span>
        <div style={{
          padding: '8px',
          borderRadius: '8px',
          background: 'rgba(0, 229, 255, 0.08)',
          color: 'var(--teal-primary)',
        }}>
          <Icon size={18} />
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'baseline', gap: '6px', marginBottom: '6px' }}>
        <span style={{ fontSize: '2rem', fontWeight: 800, color: '#F8FAFC', fontFamily: 'var(--font-mono)' }}>
          {value}
        </span>
        {unit && (
          <span style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', fontWeight: 500 }}>
            {unit}
          </span>
        )}
      </div>

      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: '4px' }}>
        {subtitle && (
          <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
            {subtitle}
          </span>
        )}
        {badge && (
          <span className={`badge badge-${badge.variant}`} style={{ fontSize: '0.65rem' }}>
            {badge.text}
          </span>
        )}
      </div>
    </div>
  );
};
