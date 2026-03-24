import type { Metadata } from "next";
import "./globals.css";
import Sidebar from "@/components/Sidebar";

export const metadata: Metadata = {
  title: "LectoAI",
  description: "Learn fun anywhere and anytime without any time limit",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ko">
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="flex gap-[12px] relative p-[12px] min-h-screen overflow-hidden text-[#1d1d1f] bg-[#ebebf0]">
        <Sidebar />
        {children}
      </body>
    </html>
  );
}
