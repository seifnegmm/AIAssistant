import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI Assistant",
  description: "Your intelligent AI companion with voice and visual capabilities",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-gray-950 text-gray-100 antialiased">
        {children}
      </body>
    </html>
  );
}
