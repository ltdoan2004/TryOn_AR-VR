import type { Metadata, Viewport } from "next";
import "./globals.css";
import Swap from "@/components/Swap";

export const metadata: Metadata = {
  title: "JEIFY",
  description: "Chat with JEIFY Chatbot",
  manifest: "/manifest.json",
  metadataBase: new URL("https://jeifypro.netlify.app/"),
  openGraph: {
    type: "website",
    url: "https://jeifypro.netlify.app/",
    title: "JEIFY Chatbot",

    description: "Chat with Jeify chatbot",
    images: [
      {
        url: "/android-chrome-192x192.png",
        width: 192,
        height: 192,
        alt: "GeminiPRO Chat AI",
      },
      {
        url: "/android-chrome-512x512.png",
        width: 512,
        height: 512,
        alt: "GeminiPRO Chat AI",
      },
    ],
  },
  icons: {
    icon: "/favicon-32x32.png",
    apple: "/apple-touch-icon.png",
  },
};
export const viewport: Viewport = {
  themeColor: "#5261ea",
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
  userScalable: true,
};
export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <Swap />
        {children}
      </body>
    </html>
  );
}
