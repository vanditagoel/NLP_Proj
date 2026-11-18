import { BarChart3, Cpu, Database, FlaskConical, Globe2, Home } from 'lucide-react';
import type { NavTab } from '../types';

interface NavbarProps {
  activeTab: NavTab;
  onSelect: (tab: NavTab) => void;
}

const ITEMS: Array<{ id: NavTab; label: string; icon: typeof Home }> = [
  { id: 'home', label: 'Home', icon: Home },
  { id: 'testing', label: 'Model Testing', icon: FlaskConical },
  { id: 'models', label: 'Models', icon: BarChart3 },
  { id: 'dataset', label: 'Dataset', icon: Database },
];

export function Navbar({ activeTab, onSelect }: NavbarProps) {
  return (
    <header className="sticky top-0 z-30 border-b border-slate-200 bg-white/95 backdrop-blur-sm">
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 py-3 sm:px-6">
        <button
          type="button"
          onClick={() => onSelect('home')}
          className="flex items-center gap-2.5 text-left"
        >
          <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-teal-700 text-white">
            <Globe2 className="h-5 w-5" />
          </span>
          <div>
            <p className="font-semibold tracking-tight text-slate-900 text-sm sm:text-base">
              IndicClimate AI
            </p>
            <p className="hidden text-xs text-slate-500 sm:block">
              CSET 346 · Hindi &amp; Bengali Stance Detection
            </p>
          </div>
        </button>

        <nav className="flex items-center gap-1 overflow-x-auto">
          {ITEMS.map((item) => {
            const Icon = item.icon;
            const active = activeTab === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => onSelect(item.id)}
                className={`inline-flex items-center gap-1.5 rounded-md px-3 py-2 text-sm font-medium whitespace-nowrap transition-colors ${
                  active
                    ? 'bg-teal-700 text-white'
                    : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900'
                }`}
              >
                <Icon className="h-4 w-4" />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        <div className="hidden items-center gap-1.5 text-xs text-slate-500 lg:flex">
          <Cpu className="h-3.5 w-3.5 text-teal-700" />
          <span>Indic &amp; Multilingual Transformers</span>
        </div>
      </div>
    </header>
  );
}
