---
name: grocery-prices
description: A skill to calculate grocery prices with country-specific taxes.
---
# Grocery Prices Skill

This skill provides grocery prices with country-specific taxes.

## Instructions
1. When asked about grocery prices or totals for a specific country, check if there is a resource file for the country in the `assets/` directory.
2. The file will be named `prices_<country_code>.json`.
3. Apply taxes to specific categories based on the country:
   - For **US** (us): Add a 20% tax to items in the **"electronics"** and **"alcohol"** categories.
   - For **Poland** (pl): Add a 23% tax to items in the **"electronics"** and **"alcohol"** categories.
4. Calculate the final price including tax and report it to the user.
