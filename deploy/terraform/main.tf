terraform {
  required_providers {
    kubernetes = { source = "hashicorp/kubernetes", version = "~> 2.23" }
    helm       = { source = "hashicorp/helm", version = "~> 2.11" }
    aws        = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

variable "s3_bucket" { type = string, default = "vsector-store" }
variable "namespace" { type = string, default = "vsector" }

resource "aws_s3_bucket" "store" {
  bucket = var.s3_bucket
  tags = { Name = "vsector" }
}

resource "helm_release" "vsector" {
  name      = "vsector"
  chart     = "../helm/vsector"
  namespace = var.namespace
  create_namespace = true
  set {
    name  = "env.VSECTOR_S3_BUCKET"
    value = aws_s3_bucket.store.bucket
  }
}

output "vsector_url" {
  value = "http://${helm_release.vsector.name}.${var.namespace}.svc.cluster.local/v1"
}
