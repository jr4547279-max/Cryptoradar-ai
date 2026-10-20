# CryptoRadar AI

Mobile-first cryptocurrency paper-trading prototype. **No real-money trading.**

## Current prototype

The existing V2 prototype is a static `index.html` file. Upload it to the repository root as `index.html` to add the app. The original ZIP is available in the ChatGPT conversation.

Features: £1,000 simulated portfolio; conservative, balanced, and aggressive profiles; CoinGecko price requests; simulated orders with fees/slippage; portfolio metrics; decision log and CSV export; emergency stop.

## Limitations

This is experimental software, not investment advice. Browser-based paper trading only runs while the page is open. Data fetching may be rate-limited; results and risk controls have not been independently validated. No exchange accounts, funds, or real-money trades are connected.

## Deployment

Host `index.html` on a static host (for example GitHub Pages if visibility/privacy requirements permit). Do not expose API secrets in client-side code. A production always-on bot requires a backend, authentication, persistent storage, exchange security and regulatory review.
