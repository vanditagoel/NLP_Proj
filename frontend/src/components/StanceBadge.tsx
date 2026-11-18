import { CheckCircle2, MinusCircle, XCircle } from 'lucide-react';
import type { StanceLabel } from '../types';

export function StanceBadge({ stance }: { stance: StanceLabel | string }) {
  if (stance === 'Favour') {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-md border border-emerald-200 bg-emerald-50 px-2.5 py-1 text-xs font-semibold text-emerald-800">
        <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600" />
        Favour
      </span>
    );
  }
  if (stance === 'Against') {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-md border border-rose-200 bg-rose-50 px-2.5 py-1 text-xs font-semibold text-rose-800">
        <XCircle className="h-3.5 w-3.5 text-rose-600" />
        Against
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1.5 rounded-md border border-slate-200 bg-slate-100 px-2.5 py-1 text-xs font-semibold text-slate-700">
      <MinusCircle className="h-3.5 w-3.5 text-slate-500" />
      None
    </span>
  );
}
