examples =[
     {
         "input": "How many policies are inforce for Q1 2024",
         "query": "SELECT count(*) FROM policies WHERE start_date >= '2024-01-01' "},
     {
         "input": "Get the policy with the highest loss",
         "query": "SELECT p.policy_id, SUM(claim_amount) as ClaimAmount from policies p join claims c on p.policy_id = c.policy_id group by p.policy_id order by SUM(claim_amount) desc limit 1"   
     }]
