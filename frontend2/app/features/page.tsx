'use client';

import Link from 'next/link';

export default function FeaturesPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
      <div className="container mx-auto px-4 py-12">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-2xl font-bold">Important Features</h1>
          <Link href="/" className="text-sm text-primary-600">Back</Link>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-4 bg-white dark:bg-gray-800 rounded shadow">
            <h4 className="font-semibold">CMRR</h4>
            <p className="text-sm text-gray-500">Coal Mine Roof Rating — measures roof competency.</p>
          </div>
          <div className="p-4 bg-white dark:bg-gray-800 rounded shadow">
            <h4 className="font-semibold">PRSUP</h4>
            <p className="text-sm text-gray-500">Percentage of roof support — indicates roof reinforcement level.</p>
          </div>
          <div className="p-4 bg-white dark:bg-gray-800 rounded shadow">
            <h4 className="font-semibold">Depth of Cover</h4>
            <p className="text-sm text-gray-500">Depth above the seam — affects stress and instability on the roof.</p>
          </div>
        </div>
      </div>
    </div>
  );
}
