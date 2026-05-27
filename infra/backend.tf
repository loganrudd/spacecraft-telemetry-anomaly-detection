# Local state — single operator, single environment.
# State file is gitignored; back it up manually (cp terraform.tfstate ~/backups/).
# To migrate to GCS later: add a bucket resource, run `terraform init -migrate-state`.
terraform {
  backend "local" {
    path = "terraform.tfstate"
  }
}
