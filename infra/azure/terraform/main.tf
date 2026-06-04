# ============================================================
# Azure Infrastructure — Multi-Agent System
# ============================================================
# Deploy:   cd infra/azure/terraform && terraform init && terraform apply
# Destroy:  terraform destroy
#
# What this provisions:
#   - Resource Group (everything lives here)
#   - Virtual Network + subnets (Container Apps managed env, PostgreSQL, Service Bus)
#   - Azure Container Apps Environment (managed runtime, equiv. to ECS cluster)
#   - Two Container Apps:
#       api:    FastAPI (min 1, max 10 — scales on HTTP RPS)
#       worker: Service Bus consumer (min 1, max 20 — scales on queue depth via KEDA)
#   - Azure Container Registry (equiv. to ECR)
#   - Service Bus Namespace (Premium tier — sessions required for FIFO)
#       Queues: task-inbox, result-outbox (sessions=true), dead-letter automatic
#   - Azure Database for PostgreSQL Flexible Server (HA, zone-redundant)
#   - Key Vault for secrets (API keys, DB password)
#   - Managed Identity for Container Apps → Key Vault + Service Bus (no credentials in code)
#   - Azure Monitor + Application Insights
#   - Log Analytics Workspace

terraform {
  required_version = ">= 1.7.0"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.97"
    }
  }
  backend "azurerm" {
    resource_group_name  = "tf-state-rg"
    storage_account_name = "yourtfstateaccount"
    container_name       = "tfstate"
    key                  = "multi-agent/prod/terraform.tfstate"
  }
}

provider "azurerm" {
  features {
    key_vault {
      purge_soft_delete_on_destroy    = false
      recover_soft_deleted_key_vaults = true
    }
  }
}

# ── Variables ──────────────────────────────────────────────────────────────

variable "location"      { default = "eastus2" }
variable "environment"   { default = "prod" }
variable "project"       { default = "multiagent" }
variable "db_password"   { sensitive = true }
variable "anthropic_key" { sensitive = true }
variable "tavily_key"    { sensitive = true; default = "" }

locals {
  prefix = "${var.project}-${var.environment}"
  tags   = {
    Project     = var.project
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

data "azurerm_client_config" "current" {}

# ── Resource Group ─────────────────────────────────────────────────────────

resource "azurerm_resource_group" "main" {
  name     = "${local.prefix}-rg"
  location = var.location
  tags     = local.tags
}

# ── Virtual Network ────────────────────────────────────────────────────────

resource "azurerm_virtual_network" "main" {
  name                = "${local.prefix}-vnet"
  resource_group_name = azurerm_resource_group.main.name
  location            = var.location
  address_space       = ["10.0.0.0/16"]
  tags                = local.tags
}

resource "azurerm_subnet" "container_apps" {
  name                 = "container-apps"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = ["10.0.0.0/23"]   # /23 min for Container Apps workload profile
  delegation {
    name = "container-apps"
    service_delegation {
      name    = "Microsoft.App/environments"
      actions = ["Microsoft.Network/virtualNetworks/subnets/join/action"]
    }
  }
}

resource "azurerm_subnet" "postgres" {
  name                 = "postgres"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = ["10.0.4.0/24"]
  service_endpoints    = ["Microsoft.Storage"]
  delegation {
    name = "postgres"
    service_delegation {
      name    = "Microsoft.DBforPostgreSQL/flexibleServers"
      actions = ["Microsoft.Network/virtualNetworks/subnets/join/action"]
    }
  }
}

# ── Log Analytics + Application Insights ──────────────────────────────────

resource "azurerm_log_analytics_workspace" "main" {
  name                = "${local.prefix}-logs"
  resource_group_name = azurerm_resource_group.main.name
  location            = var.location
  sku                 = "PerGB2018"
  retention_in_days   = 30
  tags                = local.tags
}

resource "azurerm_application_insights" "main" {
  name                = "${local.prefix}-appinsights"
  resource_group_name = azurerm_resource_group.main.name
  location            = var.location
  workspace_id        = azurerm_log_analytics_workspace.main.id
  application_type    = "web"
  tags                = local.tags
}

# ── Azure Container Registry ───────────────────────────────────────────────

resource "azurerm_container_registry" "main" {
  name                = "${var.project}${var.environment}acr"   # globally unique, alphanumeric only
  resource_group_name = azurerm_resource_group.main.name
  location            = var.location
  sku                 = "Standard"
  admin_enabled       = false   # use managed identity for pull, not admin credentials
  tags                = local.tags
}

# ── Managed Identity for Container Apps ───────────────────────────────────

resource "azurerm_user_assigned_identity" "app" {
  name                = "${local.prefix}-identity"
  resource_group_name = azurerm_resource_group.main.name
  location            = var.location
  tags                = local.tags
}

# Grant identity: pull images from ACR
resource "azurerm_role_assignment" "acr_pull" {
  scope                = azurerm_container_registry.main.id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_user_assigned_identity.app.principal_id
}

# Grant identity: send/receive on Service Bus
resource "azurerm_role_assignment" "servicebus_data" {
  scope                = azurerm_servicebus_namespace.main.id
  role_definition_name = "Azure Service Bus Data Owner"
  principal_id         = azurerm_user_assigned_identity.app.principal_id
}

# Grant identity: read secrets from Key Vault
resource "azurerm_key_vault_access_policy" "app" {
  key_vault_id = azurerm_key_vault.main.id
  tenant_id    = data.azurerm_client_config.current.tenant_id
  object_id    = azurerm_user_assigned_identity.app.principal_id
  secret_permissions = ["Get", "List"]
}

# ── Key Vault ──────────────────────────────────────────────────────────────

resource "azurerm_key_vault" "main" {
  name                = "${local.prefix}-kv"   # globally unique
  resource_group_name = azurerm_resource_group.main.name
  location            = var.location
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name            = "standard"
  soft_delete_retention_days = 7
  purge_protection_enabled   = true
  tags                = local.tags
}

# Access for Terraform to write secrets during deploy
resource "azurerm_key_vault_access_policy" "terraform" {
  key_vault_id = azurerm_key_vault.main.id
  tenant_id    = data.azurerm_client_config.current.tenant_id
  object_id    = data.azurerm_client_config.current.object_id
  secret_permissions = ["Get", "List", "Set", "Delete", "Purge"]
}

resource "azurerm_key_vault_secret" "anthropic_key" {
  name         = "anthropic-api-key"
  value        = var.anthropic_key
  key_vault_id = azurerm_key_vault.main.id
  depends_on   = [azurerm_key_vault_access_policy.terraform]
}

resource "azurerm_key_vault_secret" "db_password" {
  name         = "db-password"
  value        = var.db_password
  key_vault_id = azurerm_key_vault.main.id
  depends_on   = [azurerm_key_vault_access_policy.terraform]
}

resource "azurerm_key_vault_secret" "tavily_key" {
  name         = "tavily-api-key"
  value        = var.tavily_key
  key_vault_id = azurerm_key_vault.main.id
  depends_on   = [azurerm_key_vault_access_policy.terraform]
}

# ── Service Bus (Premium — required for sessions/FIFO) ────────────────────

resource "azurerm_servicebus_namespace" "main" {
  name                = "${local.prefix}-sb"
  resource_group_name = azurerm_resource_group.main.name
  location            = var.location
  sku                 = "Premium"
  capacity            = 1      # messaging units; scale up for higher throughput
  local_auth_enabled  = false  # managed identity only — no connection strings

  network_rule_set {
    default_action                = "Deny"
    trusted_services_allowed      = true
    public_network_access_enabled = false   # VNet only
    network_rules {
      subnet_id = azurerm_subnet.container_apps.id
      ignore_missing_vnet_service_endpoint = false
    }
  }
  tags = local.tags
}

resource "azurerm_servicebus_queue" "task_inbox" {
  name                                    = "task-inbox"
  namespace_id                            = azurerm_servicebus_namespace.main.id
  requires_session                        = true    # FIFO per SessionId (user_id)
  max_delivery_count                      = 3       # after 3 → DLQ (equiv. SQS maxReceiveCount)
  lock_duration                           = "PT5M"  # 5 min lock; worker renews every 50s
  message_time_to_live                    = "P1D"   # unprocessed messages expire after 1 day
  dead_lettering_on_message_expiration    = true
  enable_batched_operations               = true
  max_size_in_megabytes                   = 5120    # 5 GB (Premium supports up to 80 GB)
}

resource "azurerm_servicebus_queue" "result_outbox" {
  name                 = "result-outbox"
  namespace_id         = azurerm_servicebus_namespace.main.id
  requires_session     = true    # FIFO per user in result queue too
  max_delivery_count   = 5
  lock_duration        = "PT30S"
  message_time_to_live = "PT1H"  # results expire after 1 hour
}

# ── PostgreSQL Flexible Server ─────────────────────────────────────────────

resource "azurerm_private_dns_zone" "postgres" {
  name                = "${local.prefix}-postgres.private.postgres.database.azure.com"
  resource_group_name = azurerm_resource_group.main.name
  tags                = local.tags
}

resource "azurerm_private_dns_zone_virtual_network_link" "postgres" {
  name                  = "postgres-vnet-link"
  resource_group_name   = azurerm_resource_group.main.name
  private_dns_zone_name = azurerm_private_dns_zone.postgres.name
  virtual_network_id    = azurerm_virtual_network.main.id
}

resource "azurerm_postgresql_flexible_server" "main" {
  name                   = "${local.prefix}-postgres"
  resource_group_name    = azurerm_resource_group.main.name
  location               = var.location
  version                = "16"
  delegated_subnet_id    = azurerm_subnet.postgres.id
  private_dns_zone_id    = azurerm_private_dns_zone.postgres.id
  administrator_login    = "pgadmin"
  administrator_password = var.db_password

  # Zone-redundant HA — standby in a different AZ; auto-failover in ~60s
  high_availability {
    mode                      = "ZoneRedundant"
    standby_availability_zone = "2"
  }

  zone = "1"

  storage_mb   = 32768    # 32 GB; auto-grow enabled
  storage_tier = "P30"    # premium SSD

  sku_name = "GP_Standard_D2s_v3"   # 2 vCores, 8 GB RAM — good for checkpoint workload

  backup_retention_days        = 7
  geo_redundant_backup_enabled = true

  tags       = local.tags
  depends_on = [azurerm_private_dns_zone_virtual_network_link.postgres]

  lifecycle { prevent_destroy = true }
}

resource "azurerm_postgresql_flexible_server_database" "agents" {
  name      = "agents"
  server_id = azurerm_postgresql_flexible_server.main.id
  charset   = "UTF8"
  collation = "en_US.utf8"
}

resource "azurerm_postgresql_flexible_server_configuration" "max_connections" {
  name      = "max_connections"
  server_id = azurerm_postgresql_flexible_server.main.id
  value     = "200"
}

# ── Container Apps Environment ─────────────────────────────────────────────

resource "azurerm_container_app_environment" "main" {
  name                       = "${local.prefix}-env"
  resource_group_name        = azurerm_resource_group.main.name
  location                   = var.location
  log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id
  infrastructure_subnet_id   = azurerm_subnet.container_apps.id

  # Workload profile: Consumption (serverless, scale to zero) for API
  # and Dedicated D4 for workers (consistent 4 vCPU per replica)
  workload_profile {
    name                  = "Consumption"
    workload_profile_type = "Consumption"
    minimum_count         = 0
    maximum_count         = 10
  }
  workload_profile {
    name                  = "worker-profile"
    workload_profile_type = "D4"
    minimum_count         = 1
    maximum_count         = 20
  }

  tags = local.tags
}

locals {
  db_url = "postgresql://pgadmin:${var.db_password}@${azurerm_postgresql_flexible_server.main.fqdn}/agents?sslmode=require"
  sb_ns  = azurerm_servicebus_namespace.main.name
}

# ── Container App: API ─────────────────────────────────────────────────────

resource "azurerm_container_app" "api" {
  name                         = "${local.prefix}-api"
  container_app_environment_id = azurerm_container_app_environment.main.id
  resource_group_name          = azurerm_resource_group.main.name
  revision_mode                = "Single"

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.app.id]
  }

  registry {
    server   = azurerm_container_registry.main.login_server
    identity = azurerm_user_assigned_identity.app.id
  }

  ingress {
    external_enabled = true
    target_port      = 8000
    traffic_weight {
      percentage      = 100
      latest_revision = true
    }
    # Built-in WAF via Azure Front Door (attach separately) or Application Gateway
    custom_domain {
      # configure your custom domain + TLS cert here
    }
  }

  template {
    container {
      name   = "api"
      image  = "${azurerm_container_registry.main.login_server}/multi-agent-api:latest"
      cpu    = 0.5
      memory = "1Gi"

      command = ["uvicorn", "api.azure_main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

      env {
        name  = "POSTGRES_URL"
        value = local.db_url
      }
      env {
        name  = "AZURE_SERVICE_BUS_NAMESPACE"
        value = "${local.sb_ns}.servicebus.windows.net"
      }
      env {
        name  = "AZURE_TASK_QUEUE_NAME"
        value = azurerm_servicebus_queue.task_inbox.name
      }
      env {
        name  = "AZURE_RESULT_QUEUE_NAME"
        value = azurerm_servicebus_queue.result_outbox.name
      }
      env {
        name  = "AZURE_CLIENT_ID"              # tells DefaultAzureCredential which managed identity
        value = azurerm_user_assigned_identity.app.client_id
      }
      env {
        name  = "APPLICATIONINSIGHTS_CONNECTION_STRING"
        value = azurerm_application_insights.main.connection_string
      }
      # Key Vault secret references — Azure injects these as env vars at runtime
      env {
        name        = "ANTHROPIC_API_KEY"
        secret_name = "anthropic-api-key"
      }
    }

    # HTTP scaling: 1 replica per 10 concurrent requests
    min_replicas = 1
    max_replicas = 10

    http_scale_rule {
      name                = "http-scale"
      concurrent_requests = "10"
    }
  }

  secret {
    name                = "anthropic-api-key"
    key_vault_secret_id = azurerm_key_vault_secret.anthropic_key.id
    identity            = azurerm_user_assigned_identity.app.id
  }

  tags       = local.tags
  depends_on = [azurerm_role_assignment.acr_pull, azurerm_role_assignment.servicebus_data]
}

# ── Container App: Worker ──────────────────────────────────────────────────

resource "azurerm_container_app" "worker" {
  name                         = "${local.prefix}-worker"
  container_app_environment_id = azurerm_container_app_environment.main.id
  resource_group_name          = azurerm_resource_group.main.name
  revision_mode                = "Single"

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.app.id]
  }

  registry {
    server   = azurerm_container_registry.main.login_server
    identity = azurerm_user_assigned_identity.app.id
  }

  template {
    container {
      name   = "worker"
      image  = "${azurerm_container_registry.main.login_server}/multi-agent-worker:latest"
      cpu    = 2.0      # workers need more CPU for concurrent LLM calls
      memory = "4Gi"

      command = ["python", "-m", "worker.servicebus_worker"]

      env {
        name  = "POSTGRES_URL"
        value = local.db_url
      }
      env {
        name  = "AZURE_SERVICE_BUS_NAMESPACE"
        value = "${local.sb_ns}.servicebus.windows.net"
      }
      env {
        name  = "AZURE_TASK_QUEUE_NAME"
        value = azurerm_servicebus_queue.task_inbox.name
      }
      env {
        name  = "AZURE_RESULT_QUEUE_NAME"
        value = azurerm_servicebus_queue.result_outbox.name
      }
      env {
        name  = "AZURE_CLIENT_ID"
        value = azurerm_user_assigned_identity.app.client_id
      }
      env {
        name        = "ANTHROPIC_API_KEY"
        secret_name = "anthropic-api-key"
      }
      env {
        name  = "APPLICATIONINSIGHTS_CONNECTION_STRING"
        value = azurerm_application_insights.main.connection_string
      }
    }

    # KEDA: scale on Service Bus queue active message count
    # KEDA is built into Container Apps — no separate deployment needed
    min_replicas = 1
    max_replicas = 20

    custom_scale_rule {
      name             = "servicebus-queue-depth"
      custom_rule_type = "azure-servicebus"
      metadata = {
        namespace    = azurerm_servicebus_namespace.main.name
        queueName    = azurerm_servicebus_queue.task_inbox.name
        messageCount = "5"   # 1 new replica per 5 queued messages
      }
      authentication {
        secret_name       = "servicebus-connection"
        trigger_parameter = "connection"
      }
    }
  }

  secret {
    name                = "anthropic-api-key"
    key_vault_secret_id = azurerm_key_vault_secret.anthropic_key.id
    identity            = azurerm_user_assigned_identity.app.id
  }

  # KEDA needs a connection string for the Service Bus trigger
  # (managed identity doesn't work for KEDA triggers yet — use namespace-level SAS)
  secret {
    name  = "servicebus-connection"
    value = "Endpoint=sb://${local.sb_ns}.servicebus.windows.net/;SharedAccessKeyName=RootManageSharedAccessKey;SharedAccessKey=${azurerm_servicebus_namespace.main.default_primary_key}"
  }

  tags       = local.tags
  depends_on = [azurerm_role_assignment.acr_pull, azurerm_role_assignment.servicebus_data]
}

# ── Azure Monitor alerts ───────────────────────────────────────────────────

resource "azurerm_monitor_action_group" "alerts" {
  name                = "${local.prefix}-alerts"
  resource_group_name = azurerm_resource_group.main.name
  short_name          = "AgentAlerts"

  email_receiver {
    name          = "ops"
    email_address = "ops@yourdomain.com"
  }
}

resource "azurerm_monitor_metric_alert" "servicebus_dead_letter" {
  name                = "${local.prefix}-dlq-alert"
  resource_group_name = azurerm_resource_group.main.name
  scopes              = [azurerm_servicebus_namespace.main.id]
  description         = "Messages in dead-letter queue — agent tasks are failing"
  severity            = 1

  criteria {
    metric_namespace = "Microsoft.ServiceBus/namespaces"
    metric_name      = "DeadletteredMessages"
    aggregation      = "Total"
    operator         = "GreaterThan"
    threshold        = 0
    dimension {
      name     = "EntityName"
      operator = "Include"
      values   = [azurerm_servicebus_queue.task_inbox.name]
    }
  }

  action {
    action_group_id = azurerm_monitor_action_group.alerts.id
  }
}

# ── Outputs ────────────────────────────────────────────────────────────────

output "api_fqdn"                 { value = azurerm_container_app.api.latest_revision_fqdn }
output "acr_login_server"         { value = azurerm_container_registry.main.login_server }
output "service_bus_namespace"    { value = azurerm_servicebus_namespace.main.name }
output "postgres_fqdn"            { value = azurerm_postgresql_flexible_server.main.fqdn }
output "key_vault_name"           { value = azurerm_key_vault.main.name }
output "app_insights_key"         { value = azurerm_application_insights.main.instrumentation_key }
output "managed_identity_id"      { value = azurerm_user_assigned_identity.app.client_id }
