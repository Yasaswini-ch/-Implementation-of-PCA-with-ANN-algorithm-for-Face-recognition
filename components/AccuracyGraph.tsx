import React, { useState, useMemo } from 'react';

const ACCURACY_DATA: [number, number][] = [
    [1, 55],
    [5, 78],
    [10, 89],
    [20, 94],
    [30, 96.5],
    [40, 97],
    [50, 97.2],
    [75, 97.5],
    [100, 97.6],
];
const MAX_K = 100;

const getAccuracyForK = (k: number): number => {
    // Find the two points to interpolate between
    for (let i = 0; i < ACCURACY_DATA.length - 1; i++) {
        const [k1, acc1] = ACCURACY_DATA[i];
        const [k2, acc2] = ACCURACY_DATA[i + 1];
        if (k >= k1 && k <= k2) {
            const percentage = (k - k1) / (k2 - k1);
            return acc1 + percentage * (acc2 - acc1);
        }
    }
    return ACCURACY_DATA[ACCURACY_DATA.length - 1][1];
};

export const AccuracyGraph: React.FC = () => {
    const [kValue, setKValue] = useState<number>(20);
    const accuracy = useMemo(() => getAccuracyForK(kValue), [kValue]);
    
    // SVG dimensions
    const width = 500;
    const height = 250;
    const padding = { top: 20, right: 20, bottom: 40, left: 50 };
    
    const xScale = (k: number) => padding.left + (k / MAX_K) * (width - padding.left - padding.right);
    const yScale = (acc: number) => height - padding.bottom - ((acc - 50) / 50) * (height - padding.top - padding.bottom);

    const pathData = ACCURACY_DATA.map(([k, acc]) => `${xScale(k)},${yScale(acc)}`).join(' L ');
    
    return (
        <div className="bg-slate-900/50 p-6 rounded-lg border border-slate-700">
            <h4 className="text-lg font-semibold text-slate-300 mb-2">Comparative Study: Accuracy vs. k Value</h4>
            <p className="text-sm text-slate-400 mb-4">
                Use the slider to change 'k' (the number of eigenfaces used). Observe how the classification accuracy changes. Typically, accuracy increases rapidly at first and then plateaus.
            </p>

            <div className="mb-6">
                 <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto">
                    {/* Y-axis */}
                    <text x={padding.left - 15} y={padding.top - 5} textAnchor="middle" fill="#94a3b8" fontSize="10" transform={`rotate(-90, ${padding.left - 15}, ${height/2})`}>Accuracy (%)</text>
                    {[50, 60, 70, 80, 90, 100].map(val => (
                        <g key={val}>
                            <line x1={padding.left} y1={yScale(val)} x2={width-padding.right} y2={yScale(val)} stroke="#475569" strokeWidth="0.5" strokeDasharray="2,2" />
                            <text x={padding.left - 8} y={yScale(val) + 3} textAnchor="end" fill="#94a3b8" fontSize="10">{val}%</text>
                        </g>
                    ))}

                    {/* X-axis */}
                    <text x={width/2} y={height - 5} textAnchor="middle" fill="#94a3b8" fontSize="10">Number of Eigenfaces (k)</text>
                     {[0, 20, 40, 60, 80, 100].map(val => (
                        <g key={val}>
                            <line x1={xScale(val)} y1={padding.top} x2={xScale(val)} y2={height - padding.bottom} stroke="#475569" strokeWidth="0.5" strokeDasharray="2,2" />
                            <text x={xScale(val)} y={height - padding.bottom + 15} textAnchor="middle" fill="#94a3b8" fontSize="10">{val}</text>
                        </g>
                    ))}
                    
                    {/* Graph line */}
                    <path d={`M ${pathData}`} fill="none" stroke="#38bdf8" strokeWidth="2" />

                     {/* Current value marker */}
                    <circle cx={xScale(kValue)} cy={yScale(accuracy)} r="4" fill="#38bdf8" stroke="white" strokeWidth="1.5" />
                </svg>
            </div>
           
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-center">
                <div className="md:col-span-2">
                    <label htmlFor="k-slider" className="block text-sm font-medium text-slate-400 mb-1">
                        Set k Value: <span className="font-bold text-white">{kValue}</span>
                    </label>
                    <input
                        id="k-slider"
                        type="range"
                        min="1"
                        max={MAX_K}
                        step="1"
                        value={kValue}
                        onChange={(e) => setKValue(parseInt(e.target.value))}
                        className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"
                        style={{'--thumb-color': '#38bdf8'} as any}
                    />
                </div>
                <div className="text-center bg-slate-800 p-3 rounded-md">
                    <p className="text-sm text-slate-400">Simulated Accuracy</p>
                    <p className="text-2xl font-bold text-sky-400">{accuracy.toFixed(1)}%</p>
                </div>
            </div>
            <style>{`
                input[type=range]::-webkit-slider-thumb {
                    -webkit-appearance: none;
                    appearance: none;
                    width: 16px;
                    height: 16px;
                    background: var(--thumb-color);
                    border-radius: 50%;
                    cursor: pointer;
                    border: 2px solid white;
                }
                input[type=range]::-moz-range-thumb {
                     width: 16px;
                    height: 16px;
                    background: var(--thumb-color);
                    border-radius: 50%;
                    cursor: pointer;
                    border: 2px solid white;
                }
            `}</style>
        </div>
    );
};
