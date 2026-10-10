import type { Metadata } from "next";
import { Be_Vietnam_Pro, Bricolage_Grotesque, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { MobileNav, Sidebar } from "@/components/Sidebar";
import { AuthProvider } from "@/components/AuthProvider";

const bricolage = Bricolage_Grotesque({
  variable: "--font-bricolage",
  subsets: ["latin", "vietnamese"],
  weight: ["500", "600", "700"],
});

// Be Vietnam Pro: drawn for Vietnamese, so stacked diacritics stay clean.
const beVietnam = Be_Vietnam_Pro({
  variable: "--font-be-vietnam",
  subsets: ["latin", "vietnamese"],
  weight: ["400", "500", "600", "700"],
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-jetbrains-mono",
  subsets: ["latin", "vietnamese"],
  weight: ["400", "500", "700"],
});

export const metadata: Metadata = {
  title: "MeetingMind - Từ giọng nói thành tài liệu",
  description:
    "Tải lên video hoặc audio, xem tài liệu hình thành theo thời gian thực, rồi xuất PDF.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang="vi"
      suppressHydrationWarning
      className={`${bricolage.variable} ${beVietnam.variable} ${jetbrainsMono.variable} h-full antialiased`}
    >
      <body suppressHydrationWarning className="min-h-full">
        <AuthProvider>
          <a
            href="#main"
            className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-50 focus:rounded-lg focus:bg-ink focus:px-4 focus:py-2 focus:text-sm focus:text-paper"
          >
            Bỏ qua đến nội dung
          </a>
          <div className="flex min-h-dvh flex-col md:flex-row">
            <Sidebar />
            <MobileNav />
            <main id="main" className="flex-1 min-w-0">{children}</main>
          </div>
        </AuthProvider>
      </body>
    </html>
  );
}
