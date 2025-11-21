
'use client';

import Link from 'next/link';
import { config } from '@/config';

export default function Home() {
  return (
    <div className="mine-intel min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
      <div className="container mx-auto px-4 py-8">
        {/* Header / Welcome Dashboard */}
        <header className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2">
            ⛏️ {config.appTitle}
          </h1>
          <p className="text-gray-600 dark:text-gray-300 mb-6">{config.appDescription}</p>

          {/* Dashboard hero with quick actions */}
          <div className="max-w-3xl mx-auto bg-white dark:bg-gray-800/60 rounded-xl shadow-md p-6 flex flex-col md:flex-row items-center gap-4">
            <div className="flex-1 text-left">
              <h2 className="text-xl font-semibold text-gray-900 dark:text-white">Welcome</h2>
              <p className="text-sm text-gray-500 dark:text-gray-300">Use the quick actions below to explore predictions, important features, and data visualizations.</p>
            </div>
            <div className="flex gap-3">
              <Link href="/predict" className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-md inline-block text-center">
                Predict RFR
              </Link>
              <Link href="/features" className="px-4 py-2 bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-100 rounded-md inline-block text-center">
                Important Features
              </Link>
              <Link href="/graphs" className="px-4 py-2 bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-100 rounded-md inline-block text-center">
                Graph Analysis
              </Link>
            </div>
          </div>
        </header>
      </div>
    </div>
  );
}