import { dirname } from "path";
import { fileURLToPath } from "url";
import { FlatCompat } from "@eslint/eslintrc";
import stylistic from "@stylistic/eslint-plugin";
import eslintPluginBetterTailwindcss from "eslint-plugin-better-tailwindcss";
import eslintParserTypeScript from "@typescript-eslint/parser";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const compat = new FlatCompat({ baseDirectory: __dirname, });

const eslintConfig = [
    ...compat.extends("next/core-web-vitals", "next/typescript"),
    {
        files: ["**/*.{ts,tsx,cts,mts}"],
        languageOptions: {
            parser: eslintParserTypeScript,
            parserOptions: { project: true }
        }
    },
    {
        files: ["**/*.{jsx,tsx}"],
        languageOptions: { parserOptions: { ecmaFeatures: { jsx: true } } },
        plugins: { "better-tailwindcss": eslintPluginBetterTailwindcss },
        rules: {
            // enable all recommended rules to report a warning
            ...eslintPluginBetterTailwindcss.configs["recommended-warn"].rules,
            // enable all recommended rules to report an error
            ...eslintPluginBetterTailwindcss.configs["recommended-error"].rules,
            "better-tailwindcss/no-unregistered-classes": 0,
            "better-tailwindcss/enforce-consistent-line-wrapping": 0,
        },
        settings: {
            "better-tailwindcss": {
                // tailwindcss 4: the path to the entry file of the css based tailwind config (eg: `src/global.css`)
                entryPoint: "src/global.css",
            }
        }
    },
    {
        plugins: { "@stylistic": stylistic },
        rules: {
            "@stylistic/array-element-newline": ["error", "consistent"],
            "@stylistic/brace-style": ["error", "stroustrup", { allowSingleLine: true }],
            "@stylistic/function-call-argument-newline": ["error", "consistent"],
            "@stylistic/function-paren-newline": ["error", { minItems: 4 }],
            "@stylistic/object-curly-spacing": ["error", "always"],
            "@stylistic/multiline-comment-style": ["error", "separate-lines"],
            "@stylistic/multiline-ternary": ["error", "always-multiline"],
            "@stylistic/object-curly-newline": ["error", { multiline: true }],
            "@stylistic/padded-blocks": ["error", "never"],
            "@stylistic/quote-props": ["error", "as-needed"],

            "@stylistic/array-bracket-newline": 2,
            "@stylistic/array-bracket-spacing": 2,
            "@stylistic/arrow-parens": 2,
            "@stylistic/arrow-spacing": 2,
            "@stylistic/block-spacing": 2,
            "@stylistic/comma-dangle": 0,
            "@stylistic/comma-spacing": 2,
            "@stylistic/comma-style": 2,
            "@stylistic/computed-property-spacing": 2,
            "@stylistic/dot-location": 2,
            "@stylistic/eol-last": 2,
            "@stylistic/function-call-spacing": 2,
            "@stylistic/generator-star-spacing": 2,
            "@stylistic/indent": ["error", 4],
            "@stylistic/implicit-arrow-linebreak": 2,
            "@stylistic/jsx-quotes": 2,
            "@stylistic/key-spacing": 2,
            "@stylistic/keyword-spacing": 2,
            "@stylistic/lines-around-comment": 2,
            "@stylistic/lines-between-class-members": 2,
            "@stylistic/new-parens": 2,
            "@stylistic/newline-per-chained-call": 2,
            "@stylistic/no-confusing-arrow": 2,
            "@stylistic/no-extra-parens": 2,
            "@stylistic/no-extra-semi": 2,
            "@stylistic/no-floating-decimal": 2,
            "@stylistic/no-multi-spaces": 2,
            "@stylistic/no-multiple-empty-lines": 2,
            "@stylistic/no-trailing-spaces": 2,
            "@stylistic/no-whitespace-before-property": 2,
            "@stylistic/nonblock-statement-body-position": 2,
            "@stylistic/object-property-newline": 2,
            "@stylistic/one-var-declaration-per-line": 2,
            "@stylistic/operator-linebreak": 2,
            "@stylistic/padding-line-between-statements": 2,
            "@stylistic/quotes": 2,
            "@stylistic/rest-spread-spacing": 2,
            "@stylistic/semi": 2,
            "@stylistic/semi-spacing": 2,
            "@stylistic/semi-style": 2,
            "@stylistic/space-before-blocks": 2,
            "@stylistic/space-before-function-paren": 2,
            "@stylistic/space-in-parens": 2,
            "@stylistic/space-infix-ops": 2,
            "@stylistic/space-unary-ops": 2,
            "@stylistic/spaced-comment": 2,
            "@stylistic/switch-colon-spacing": 2,
            "@stylistic/template-curly-spacing": 2,
            "@stylistic/template-tag-spacing": 2,
            "@stylistic/wrap-iife": 2,
            "@stylistic/wrap-regex": 2,
            "@stylistic/yield-star-spacing": 2,
            "@next/next/no-img-element": 0,
            // "@stylistic/linebreak-style": 2,
        }
    },
];

export default eslintConfig;
