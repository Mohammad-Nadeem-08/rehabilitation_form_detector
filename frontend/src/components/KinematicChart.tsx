import React from 'react';
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis,
  Tooltip, Legend, CartesianGrid, ReferenceLine
} from 'recharts';

interface KinematicChartProps {
  title?: string;
  data: any[];
  mode?: 'live' | 'comparison';
}

export const KinematicChart: React.FC<KinematicChartProps> = ({
  title = "Kinematic Trajectory (Right Arm Elevation & Elbow)",
  data,
  mode = 'live',
}) => {
  return (
    <div className="glass-panel" style={{ padding: '20px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
        <h4 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#F8FAFC' }}>
          {title}
        </h4>
        <span className="badge badge-teal" style={{ fontSize: '0.65rem' }}>
          {mode === 'live' ? 'Live Telemetry' : 'DTW Aligned Curves'}
        </span>
      </div>

      <div style={{ width: '100%', height: 260 }}>
        {data.length === 0 ? (
          <div style={{
            height: '100%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'var(--text-muted)',
            fontSize: '0.85rem',
            border: '1px dashed var(--border-subtle)',
            borderRadius: '8px',
          }}>
            No kinematic data points to display. Start camera tracking or record a repetition.
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 10, right: 15, left: -15, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(51, 65, 85, 0.4)" />
              <XAxis
                dataKey={mode === 'live' ? 'time' : 'step'}
                stroke="#64748B"
                fontSize={11}
                tickFormatter={(v) => (mode === 'live' ? `${v}s` : `${v}`)}
              />
              <YAxis
                domain={[0, 180]}
                stroke="#64748B"
                fontSize={11}
                ticks={[0, 30, 60, 90, 120, 150, 180]}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#0F172A',
                  borderColor: 'rgba(0, 229, 255, 0.3)',
                  borderRadius: '8px',
                  fontSize: '0.8rem',
                  color: '#F8FAFC',
                }}
              />
              <Legend wrapperStyle={{ fontSize: '0.75rem', paddingTop: '8px' }} />

              {/* Target 90° coronal plane guide line */}
              <ReferenceLine
                y={90}
                label={{ value: 'Target 90°', fill: '#10B981', fontSize: 10, position: 'right' }}
                stroke="#10B981"
                strokeDasharray="4 4"
                strokeWidth={1.5}
              />

              {mode === 'live' ? (
                <>
                  <Line
                    type="monotone"
                    dataKey="elevation"
                    name="Arm Elevation (°)"
                    stroke="#00E5FF"
                    strokeWidth={2.5}
                    dot={false}
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="elbow"
                    name="Elbow Angle (°)"
                    stroke="#F59E0B"
                    strokeWidth={1.5}
                    strokeDasharray="3 3"
                    dot={false}
                    isAnimationActive={false}
                  />
                </>
              ) : (
                <>
                  <Line
                    type="monotone"
                    dataKey="refElevation"
                    name="Reference Elevation (°)"
                    stroke="#10B981"
                    strokeWidth={2}
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="homeElevation"
                    name="Patient Elevation (°)"
                    stroke="#00E5FF"
                    strokeWidth={2.5}
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="homeElbow"
                    name="Patient Elbow (°)"
                    stroke="#F59E0B"
                    strokeWidth={1.5}
                    strokeDasharray="3 3"
                    dot={false}
                  />
                </>
              )}
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
};
