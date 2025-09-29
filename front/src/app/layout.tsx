import type { Metadata, Viewport } from "next";
import "./globals.css";

import localFont from "next/font/local";

export const metadata: Metadata = {
    title: "LCT",
    description: "",
};
export const viewport: Viewport = {
    initialScale: 1,
    width: "device-width",
    minimumScale: 1.0,
    maximumScale: 1.0,
    userScalable: false
};
const AlumniSansFont = localFont({
    src: [
        {
            path: "../../public/fonts/AlumniSans-Regular.ttf",
            weight: "400"
        }
    ],
    variable: "--font-alumni-sans"
});

const BrunoAceFont = localFont({
    src: [
        {
            path: "../../public/fonts/BrunoAce-Regular.ttf",
            weight: "400"
        }
    ],
    variable: "--font-bruno-ace"
});

export default function RootLayout ({ children, }: Readonly<{ children: React.ReactNode; }>) {
    return (
        <html lang="en">
            <body
                className={`
                  overflow-hidden
                  bg-white
                  text-white
                  antialiased
                  ${AlumniSansFont.variable}
                  ${BrunoAceFont.variable}
                `}
            >
                {children}
            </body>
        </html>
    );
}
