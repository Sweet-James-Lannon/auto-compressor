# PDF Compression API - Integration Guide

Quick guide for integrating with the SJ PDF Compressor service.

---

## Step 1: Get Your Credentials

You need two things from your Azure admin:

| Item | Example |
|------|---------|
| **API URL** | `https://sj-doc-compressor.azurewebsites.net/compress` |
| **API Token** | `your-secret-api-token-here` |

---

## Step 2: Make a Request

### The Basics

```
POST /compress
Authorization: Bearer YOUR_API_TOKEN
Content-Type: multipart/form-data

Body: pdf file
```

### cURL Example (Copy & Paste)

```bash
curl -X POST "https://sj-doc-compressor.azurewebsites.net/compress" \
  -H "Authorization: Bearer YOUR_API_TOKEN" \
  -F "pdf=@document.pdf"
```

Replace:
- `YOUR_API_TOKEN` with your actual key
- `document.pdf` with your file path

---

## Step 3: Handle the Response

### Success (200)

```json
{
  "success": true,
  "original_mb": 50.0,
  "compressed_mb": 5.2,
  "reduction_percent": 89.6,
  "compressed_pdf_b64": "JVBERi0xLjQK...",
  "compression_method": "ghostscript",
  "request_id": "a1b2c3d4"
}
```

**To get the compressed PDF:** Decode `compressed_pdf_b64` from base64.

### Errors

| Code | Meaning | Fix |
|------|---------|-----|
| 401 | Missing auth header | Add `Authorization: Bearer KEY` header |
| 403 | Wrong token | Check your token is correct |
| 400 | Invalid PDF | Make sure file is a valid PDF |
| 413 | File too big | Max size is 300MB |
| 500 | Server error | Try again or contact support |

---

## n8n Setup

### HTTP Request Node Configuration

1. **Method:** `POST`
2. **URL:** `https://sj-doc-compressor.azurewebsites.net/compress`
3. **Authentication:** Header Auth
   - Name: `Authorization`
   - Value: `Bearer YOUR_API_TOKEN`
4. **Body Content Type:** Form-Data
5. **Body Parameters:**
   - Name: `pdf`
   - Type: Binary
   - Value: Your PDF file data

### Getting the Compressed PDF in n8n

Use a **Code node** after the HTTP Request:

```javascript
// Decode the base64 PDF from the response
const compressedPdfBase64 = $input.first().json.compressed_pdf_b64;
const buffer = Buffer.from(compressedPdfBase64, 'base64');

return {
  binary: {
    data: buffer.toString('base64'),
    mimeType: 'application/pdf',
    fileName: 'compressed.pdf'
  }
};
```

---

## Salesforce Apex Example

```apex
public class PDFCompressor {

    private static final String API_URL = 'https://sj-doc-compressor.azurewebsites.net/compress';
    private static final String API_TOKEN = 'YOUR_API_TOKEN'; // Store in Custom Metadata

    @future(callout=true)
    public static void compressPDF(Id attachmentId) {
        // Get the PDF file
        Attachment att = [SELECT Body, Name FROM Attachment WHERE Id = :attachmentId];

        // Build the request
        HttpRequest req = new HttpRequest();
        req.setEndpoint(API_URL);
        req.setMethod('POST');
        req.setHeader('Authorization', 'Bearer ' + API_TOKEN);

        // Create multipart body
        String boundary = '----WebKitFormBoundary' + String.valueOf(DateTime.now().getTime());
        req.setHeader('Content-Type', 'multipart/form-data; boundary=' + boundary);

        String body = '--' + boundary + '\r\n';
        body += 'Content-Disposition: form-data; name="pdf"; filename="' + att.Name + '"\r\n';
        body += 'Content-Type: application/pdf\r\n\r\n';
        body += EncodingUtil.base64Encode(att.Body) + '\r\n';
        body += '--' + boundary + '--';

        req.setBody(body);
        req.setTimeout(120000); // 2 minute timeout for large files

        // Send request
        Http http = new Http();
        HttpResponse res = http.send(req);

        if (res.getStatusCode() == 200) {
            Map<String, Object> result = (Map<String, Object>) JSON.deserializeUntyped(res.getBody());
            String compressedBase64 = (String) result.get('compressed_pdf_b64');

            // Save compressed PDF
            Attachment compressed = new Attachment();
            compressed.Name = 'compressed_' + att.Name;
            compressed.Body = EncodingUtil.base64Decode(compressedBase64);
            compressed.ParentId = att.ParentId;
            insert compressed;
        }
    }
}
```

---

## Python Example

```python
import requests
import base64

API_URL = "https://sj-doc-compressor.azurewebsites.net/compress"
API_TOKEN = "YOUR_API_TOKEN"

def compress_pdf(file_path):
    with open(file_path, 'rb') as f:
        response = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {API_TOKEN}"},
            files={"pdf": f}
        )

    if response.status_code == 200:
        result = response.json()
        compressed_bytes = base64.b64decode(result['compressed_pdf_b64'])

        output_path = file_path.replace('.pdf', '_compressed.pdf')
        with open(output_path, 'wb') as f:
            f.write(compressed_bytes)

        print(f"Compressed: {result['original_mb']}MB -> {result['compressed_mb']}MB")
        print(f"Saved to: {output_path}")
        return output_path
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

# Usage
compress_pdf("large_document.pdf")
```

---

## JavaScript/Node.js Example

```javascript
const fs = require('fs');
const FormData = require('form-data');
const axios = require('axios');

const API_URL = 'https://sj-doc-compressor.azurewebsites.net/compress';
const API_TOKEN = 'YOUR_API_TOKEN';

async function compressPDF(filePath) {
  const form = new FormData();
  form.append('pdf', fs.createReadStream(filePath));

  const response = await axios.post(API_URL, form, {
    headers: {
      'Authorization': `Bearer ${API_TOKEN}`,
      ...form.getHeaders()
    }
  });

  if (response.data.success) {
    const compressed = Buffer.from(response.data.compressed_pdf_b64, 'base64');
    const outputPath = filePath.replace('.pdf', '_compressed.pdf');
    fs.writeFileSync(outputPath, compressed);

    console.log(`Compressed: ${response.data.original_mb}MB -> ${response.data.compressed_mb}MB`);
    return outputPath;
  }
}

// Usage
compressPDF('large_document.pdf');
```

---

## Test Your Setup

### 1. Check the service is running

```bash
curl https://sj-doc-compressor.azurewebsites.net/health
```

Should return:
```json
{"status": "healthy", "ghostscript": true, ...}
```

### 2. Test with a real PDF

```bash
curl -X POST "https://sj-doc-compressor.azurewebsites.net/compress" \
  -H "Authorization: Bearer YOUR_API_TOKEN" \
  -F "pdf=@test.pdf" \
  -o response.json

# Check the result
cat response.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Compressed: {d[\"original_mb\"]}MB -> {d[\"compressed_mb\"]}MB ({d[\"reduction_percent\"]}% reduction)')"
```

---

## Need Help?

1. Check the health endpoint first
2. Verify your API token is correct
3. Make sure your PDF file is valid (opens locally)
4. Check file size is under 300MB

Contact the Sweet James dev team for API token issues.
