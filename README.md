## Running miner in linux google cloud

1) Set up google cloud
2) run: sudo systemctl stop miner_nbeats

gcloud compute addresses create miner-live-9002 --project=local-bliss-404809 --description=34.130.178.90 --region=northamerica-northeast2 && gcloud compute instances add-access-config miner-live-9002 --project=local-bliss-404809 --zone=northamerica-northeast2-a --address=IP_OF_THE_NEWLY_CREATED_STATIC_ADDRESS

# Setting up Google Cloud
1) For each new machine --> clone from machine image
2) reserve static ip address:
gcloud compute addresses create miner-live-9002 --project=local-bliss-404809 --description=34.130.178.90 --region=northamerica-northeast2 && gcloud compute instances add-access-config miner-live-9002 --project=local-bliss-404809 --zone=northamerica-northeast2-a --address=IP_OF_THE_NEWLY_CREATED_STATIC_ADDRESS
3) If cloning disk image, make sure to disable systemd so that it doesnt impact previous machines
4) set up fire wall rules as well
5) 