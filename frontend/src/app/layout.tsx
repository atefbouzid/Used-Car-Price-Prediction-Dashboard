import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Used Car Price Prediction Dashboard',
  description: 'Professional dashboard for predicting used car prices using multiple ML models',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} bg-dark-950 text-white min-h-screen`}>
        <div className="min-h-screen bg-gradient-to-br from-dark-950 via-dark-900 to-dark-800">
          {children}
        </div>
      </body>
    </html>
  )
}
