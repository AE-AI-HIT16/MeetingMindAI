# Beautiful Frontend UI/UX Skill — MeetASR

This document outlines the visual guidelines, typography standards, and interactive UX designs for building the frontend layer in MeetASR. It provides rules to construct high-fidelity, polished, and responsive web applications.

---

## 1. Visual Design System

To capture user interest and convey a premium look, developers and AI Agents must follow clean, modern aesthetics rather than raw, default configurations.

### 1.1 Color Palettes & Sleek Dark Mode
Avoid raw primary colors (e.g., pure #ff0000, #00ff00). Instead, leverage tailored HSL or hex palettes:
- **Primary Color:** Sleek Indigo `#4f46e5` or Violet `#7c3aed`.
- **Background (Dark Mode):** Never use absolute black (`#000`). Opt for deep obsidian/navy `#0b0f19` for the body, paired with surface backgrounds of `#151b2c` for cards and modals.
- **Text:** High contrast white `#f3f4f6` (Off-white) for headings, muted gray `#9ca3af` for body text.
- **Status Badges:**
  - High Priority: Sunset Red `#ef4444`.
  - Medium Priority: Amber Yellow `#f59e0b`.
  - Low Priority: Emerald Green `#10b981`.

### 1.2 Glassmorphism & Depth
Apply modern blur backdrops to add visual layers:
- Backdrops: `backdrop-filter: blur(12px); background: rgba(21, 27, 44, 0.7);`
- Thin Borders: `border: 1px solid rgba(255, 255, 255, 0.08);`
- Soft Shadows: `box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);`

### 1.3 Typography
Import high-quality modern Google Fonts rather than native system defaults:
- Recommended: **Inter**, **Outfit**, or **Plus Jakarta Sans**.
- Utilize clean font weights (400 for regular readouts, 600-700 for page headings) to organize page hierarchy.

---

## 2. Interactive Meeting UI/UX Specifications

### 2.1 Synced Audio Timeline & Transcripts
- **Waveform UI:** Replace plain HTML5 audio players with an audio waveform timeline using libraries like `wavesurfer.js`.
- **Text Synchronization:** Highlight the corresponding sentence blocks in the transcript area while the audio is playing. Clicking a transcript block should immediately seek the audio player to the segment's start time.

### 2.2 Chatbot-Style Timeline Transcripts
- Group sentences by speaker. Assign distinct avatar icons and color codings to differentiate speakers (e.g., Speaker 0 is blue, Speaker 1 is violet).
- Place accurate, clickable timestamps next to the speaker's name (`[01:23]`).

### 2.3 Meeting Report Layout
- **Executive Summary:** Embed within a card containing a subtle linear gradient background.
- **Action Items:** Organize as tables or interactive checklist cards with color-coded priority badges and clear assignee tags.
- **Key Decisions:** Group into highlight quote cards with distinctive icons (e.g., gold checkmark) to emphasize important meeting outcomes.

---

## 3. Reference CSS & Transitions

### 3.1 Premium Hover Effects

```css
/* Custom Meeting Card */
.meeting-card {
  background: rgba(21, 27, 44, 0.75);
  backdrop-filter: blur(12px);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 16px;
  padding: 24px;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
}

/* Smooth translate on hover */
.meeting-card:hover {
  transform: translateY(-4px);
  border-color: rgba(99, 102, 241, 0.4);
  box-shadow: 0 12px 30px rgba(99, 102, 241, 0.15);
}
```

### 3.2 Pulse Loader for Processing States

```css
@keyframes pulse-glow {
  0%, 100% {
    transform: scale(1);
    box-shadow: 0 0 0 0 rgba(99, 102, 241, 0.7);
  }
  50% {
    transform: scale(1.05);
    box-shadow: 0 0 20px 8px rgba(99, 102, 241, 0);
  }
}

.processing-badge {
  background: linear-gradient(135deg, #4f46e5, #7c3aed);
  color: #ffffff;
  padding: 6px 12px;
  border-radius: 9999px;
  font-weight: 600;
  font-size: 0.85rem;
  animation: pulse-glow 2s infinite ease-in-out;
}
```

---

## 4. AI Verification Checklist

When constructing or auditing frontend components, the Agent must verify:
- [ ] Does the visual style utilize a modern dark mode theme?
- [ ] Are custom typography families imported and applied correctly?
- [ ] Do buttons and interactive cards feature micro-animations on hover states?
- [ ] Are all mock layouts populated with clean data instead of ugly lorem-ipsum placeholders?
- [ ] Does clicking transcript text blocks update the audio playback seek timeline?
- [ ] Is the page layout responsive, adapting cleanly to mobile displays?
