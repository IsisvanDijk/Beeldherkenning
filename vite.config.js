import { defineConfig } from "vite";

/** @type {import('vite').UserConfig} */

// https://vitejs.dev/config

export default defineConfig({
    base: "./",
    build: {
        outDir: "docs",
        emptyOutDir: true,
    },
});
