'use client';

import Link from 'next/link';

export default function GraphsPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
      <div className="container mx-auto px-4 py-12">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-2xl font-bold">Graph Analysis</h1>
          <Link href="/" className="text-sm text-primary-600">Back</Link>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="h-64 bg-white dark:bg-gray-800 rounded shadow flex items-center justify-center text-gray-400">Feature importance chart (placeholder)</div>
          <div className="h-64 bg-white dark:bg-gray-800 rounded shadow flex items-center justify-center text-gray-400">Distribution / correlation chart (placeholder)</div>
        </div>
      </div>
    </div>
  );
}
