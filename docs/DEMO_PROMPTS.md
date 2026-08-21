# Demo prompts

## P1 · Football attack and defence (Claude Desktop / Sonnet 5.0)

Needs: a client that can fetch web pages, and the server reachable (Claude
Desktop → stdio, so the agent uploads by copying files into the datasets
directory — `get_upload_instructions` tells it where). Tested 

> Analyse the attack and defense capabilities of the French football league
> 2025/2026 using a Bayesian Poisson Attack Defense Model.
>
> Hint: For the data, use footballdatabase.com — specifically the
> league-scores-tables page (format:
> `footballdatabase.com/league-scores-tables/france-ligue-1-2025-2026`).

Remarks: Tried with OpenCode (Qwee 3.5 397B Reasoning), it could not get the data so it simulated it w/o telling me.

## P2 · Bayesian regression (Claude Desktop / Sonnet 5.0) Looping 
>Use the 1d regression dataset. Start with a simple linear regression, then keep improving the NLPD — up to 20 iterations, stopping early if three in a row bring no improvement on the testset. Record NLPD.

