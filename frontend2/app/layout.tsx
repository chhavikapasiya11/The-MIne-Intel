import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { config } from '@/config';
import { AppStateProvider } from '@/context/AppState';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: config.appTitle,
  description: config.appDescription,
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <AppStateProvider>{children}</AppStateProvider>
      </body>
    </html>
  );
}

