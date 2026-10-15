# TaskFlow — Minimal To-Do List Chrome Extension

A clean, minimal, and attractive productivity To-Do List Chrome Extension built using **Vanilla JavaScript**, **HTML5**, and **CSS3**, adhering strictly to **Google Chrome Manifest V3**.

---

## ✨ Features

- **Minimalist Aesthetic:** Distraction-free, modern interface with smooth micro-interactions.
- **Instant Categories:** One-click tags (`General`, `College`, `Coding`, `Personal`).
- **Live Status Filtering:** Quick tabs for `All`, `Active`, and `Completed` tasks.
- **Dynamic Toolbar Badge:** Displays active pending task count directly on the Chrome toolbar extension icon.
- **Light & Dark Theme:** Minimalist dark mode toggle with automatic preference persistence.
- **Zero-Dependency & CSP Compliant:** Uses pure Vanilla JavaScript and inline SVGs—no external CDN scripts or tracker cookies.
- **Persistent Storage:** Synchronized via `chrome.storage.sync` with automatic fallback to `localStorage` when previewed in web browsers.

---

## 📁 File Structure

```
PE/Task2_ChromeExtension/
├── manifest.json        # Chrome Extension Manifest V3 configuration
├── popup.html           # Minimal UI structure
├── popup.css            # Light & Dark theme styling and animations
├── popup.js             # Core application logic (CRUD, filtering, theme)
├── background.js        # Service worker for live icon badge management
├── icons/               # Extension icons (16px, 48px, 128px)
│   ├── icon16.png
│   ├── icon48.png
│   └── icon128.png
├── generate_icons.py    # Icon generator script
├── SUBMISSION_REPORT.md # Academic documentation & list of AI prompts
└── README.md            # Installation and setup instructions
```

---

## 🛠️ How to Install in Google Chrome (Developer Mode)

1. Open **Google Chrome**.
2. Navigate to: `chrome://extensions` (or click Chrome Menu `⋮` → **Extensions** → **Manage Extensions**).
3. Toggle the **Developer mode** switch in the top-right corner to **ON**.
4. Click the **Load unpacked** button in the top-left corner.
5. In the file picker, select this extension folder:
   ```
   /Users/samarthkolarkar/Classroom/Web Tech -College/PE/Task2_ChromeExtension
   ```
6. **TaskFlow** will appear in your Chrome extensions list!
7. Click the **Extensions** (puzzle piece) icon in your Chrome toolbar and pin **TaskFlow** for 1-click access.

---

## 🧪 Quick Test & Preview

You can also test the popup immediately by opening `popup.html` directly in any web browser:
```bash
open popup.html
```
*(The extension automatically detects standard browser environments and seamlessly uses `localStorage` for testing).*
