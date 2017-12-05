.PHONY: sync

start:
	@gcloud compute instances start ml2

stop:
	@gcloud compute instances stop ml2
