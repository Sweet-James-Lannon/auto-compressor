# SJ PDF Compressor

Sweet James in-house PDF compression service for Salesforce integration.

## Purpose

This service compresses PDF files before storage in Salesforce, reducing document size by up to 90% while maintaining document integrity. It uses Ghostscript with lossless compression to optimize legal documents and case files.

## Features

- Lossless PDF compression with up to 90% file size reduction
- Bearer token authentication for secure API access
- Automatic file cleanup after processing
- Azure deployment ready
- Salesforce integration endpoints
- Health monitoring and status checks

## Setup

### Local Development

```bash
# Clone the repository
git clone https://github.com/Sweet-James-Lannon/auto-compressor.git
cd auto-compressor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
API_TOKEN=your-secure-api-token-here
DEBUG=False
PORT=5000
FILE_RETENTION_SECONDS=30
```

### Run Locally

```bash
python app.py
```

The service will start on `http://localhost:5000`

## API Endpoints

All endpoints require Bearer token authentication: `Authorization: Bearer your-api-token`

### Health Check

Check service status:

```http
GET /health
```

Response:
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "timestamp": "2024-11-19T14:30:00.000Z"
}
```

### Compress PDF

Compress a PDF file for Salesforce storage:

```http
POST /compress
Authorization: Bearer your-api-token
Content-Type: multipart/form-data
```

Request Parameters:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `pdf` | file | Yes | PDF file to compress (max 300MB) |
| `matter_id` | string | No | Matter ID for tracking |
| `user_email` | string | No | Requester email |

Success Response (200):

```json
{
  "success": true,
  "original_mb": 50.0,
  "compressed_mb": 5.2,
  "compressed_pdf_b64": "JVBERi0xLjQKJcfs...",
  "request_id": "a1b2c3d4",
  "matter_id": "MAT-2024-12345",
  "user_email": "user@sweetjames.com"
}
```

Error Responses:

- `400` - Invalid or missing PDF file
- `401` - Missing authentication
- `403` - Invalid API token
- `413` - File exceeds 300MB limit
- `500` - Compression failed

## Integration Example (cURL)

```bash
# Compress a PDF for Salesforce
curl -X POST https://your-service.azurewebsites.net/compress \
  -H "Authorization: Bearer your-api-token" \
  -F "pdf=@legal-document.pdf" \
  -F "matter_id=MAT-2024-12345" \
  -F "user_email=attorney@sweetjames.com" \
  -o response.json

# Extract compressed PDF from response
cat response.json | jq -r '.compressed_pdf_b64' | base64 -d > compressed.pdf
```

## Azure Deployment

### Prerequisites

- Azure subscription
- Azure CLI installed
- Resource group created

### Quick Deploy

```bash
# Set variables
RESOURCE_GROUP="your-resource-group"
APP_NAME="sj-pdf-compressor"
LOCATION="eastus"

# Create App Service Plan
az appservice plan create \
  --name "${APP_NAME}-plan" \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION \
  --sku P1V2 \
  --is-linux

# Create Web App
az webapp create \
  --resource-group $RESOURCE_GROUP \
  --plan "${APP_NAME}-plan" \
  --name $APP_NAME \
  --runtime "PYTHON:3.11"

# Configure environment variables
az webapp config appsettings set \
  --resource-group $RESOURCE_GROUP \
  --name $APP_NAME \
  --settings \
    API_TOKEN="your-secure-token" \
    DEBUG="False"

# Deploy code
zip -r deploy.zip . -x "*.git*" -x "venv/*" -x "__pycache__/*"

az webapp deployment source config-zip \
  --resource-group $RESOURCE_GROUP \
  --name $APP_NAME \
  --src deploy.zip

# Test deployment
curl https://${APP_NAME}.azurewebsites.net/health
```

### View Logs

```bash
az webapp log tail --resource-group $RESOURCE_GROUP --name $APP_NAME
```

## Troubleshooting

### Service Not Responding

Check the health endpoint to verify the service is running:

```bash
curl https://your-service.azurewebsites.net/health
```

If the service is down, check Azure logs:

```bash
az webapp log tail --resource-group $RESOURCE_GROUP --name $APP_NAME
```

### Authentication Failed

Verify your API token is correct and included in the header:

```bash
curl -X POST https://your-service.azurewebsites.net/compress \
  -H "Authorization: Bearer your-api-token" \
  -F "pdf=@test.pdf"
```

### Compression Failed

Ensure the PDF is a valid file and not already corrupted:

- File should be less than 300MB
- Check that the PDF can be opened locally
- Verify sufficient disk space on the server

### Low Compression Ratio

Compression effectiveness depends on the PDF content:

- Already-compressed PDFs may not compress further
- Text-heavy documents compress better than image-heavy ones
- Scanned documents typically achieve 70-90% reduction

## Support

For issues or questions:

1. Check the Azure logs: `az webapp log tail --resource-group $RESOURCE_GROUP --name $APP_NAME`
2. Review the health endpoint status
3. Test with a known working PDF file
4. Contact the Sweet James development team

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.