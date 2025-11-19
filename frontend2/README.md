# Mine-Intel Frontend (Next.js)

A modern React/Next.js frontend for the Mine-Intel roof fall risk prediction application.

## Features

- 🎯 Interactive form for mining parameters
- 💬 Natural language chat assistant
- 📊 Real-time prediction display
- 🎨 Modern UI with Tailwind CSS
- 🔄 Type-safe with TypeScript

## Getting Started

### Prerequisites

- Node.js 18+ and npm/yarn/pnpm

### Installation

1. Install dependencies:
```bash
npm install
# or
yarn install
# or
pnpm install
```

2. Set up environment variables (optional):
Create a `.env.local` file:
```
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:5000
```

3. Run the development server:
```bash
npm run dev
# or
yarn dev
# or
pnpm dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Project Structure

```
frontend2/
├── app/                    # Next.js App Router
│   ├── layout.tsx         # Root layout
│   ├── page.tsx           # Home page
│   └── globals.css        # Global styles
├── components/            # React components
│   ├── PredictionForm.tsx
│   ├── ChatAssistant.tsx
│   └── PredictionCard.tsx
├── services/              # API clients
│   └── api-client.ts
├── utils/                 # Utility functions
│   ├── nlp.ts            # NLP extraction
│   └── validators.ts     # Form validation
├── types/                 # TypeScript types
│   └── index.ts
├── config/                # Configuration
│   └── index.ts
└── public/               # Static assets
```

## Build for Production

```bash
npm run build
npm start
```

## Technologies

- **Next.js 14** - React framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **Axios** - HTTP client

