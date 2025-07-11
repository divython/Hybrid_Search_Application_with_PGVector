# Real Documents Directory

This directory is for your actual financial documents (annual reports, earnings calls, etc.).

## Instructions

1. **Copy your real documents here**: Place your .txt files in this directory
2. **Naming Convention**: Use format `CompanyName_DocumentType_Year.txt` for best results
   - Example: `Apple_AnnualReport_2023.txt`
   - Example: `Microsoft_EarningsCall_Q4_2023.txt`
   - Example: `TSLA_10K_2023.txt`

3. **Supported Document Types**:
   - Annual Reports (10-K, Annual Report)
   - Earnings Calls (Earnings Call, Conference Call)
   - Quarterly Reports (10-Q, Q1, Q2, Q3, Q4)
   - Other financial documents

4. **File Format**: 
   - Use plain text (.txt) files
   - Ensure good text quality and structure
   - Remove excessive formatting or special characters

## Processing Your Documents

After copying your files here, run the ingestion script to process them:

```bash
python scripts/ingest_documents.py --real-data
```

Or modify the ingestion script to point to this directory.

## Notes

- The system will automatically extract metadata from filenames
- Larger documents will be chunked for better search performance
- Both dense and sparse vectors will be created for each chunk
- You can mix different document types and companies

## Example File Structure

```
real_documents/
├── Apple_AnnualReport_2023.txt
├── Microsoft_EarningsCall_Q4_2023.txt
├── Google_10K_2023.txt
├── Amazon_Q3_2023_Earnings.txt
└── Tesla_AnnualReport_2023.txt
```
