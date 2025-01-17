## Visual Retrieval - Open-Webui - ColQwen2 / Vespa

## visual retrieval via ColQwen2 (over infinity embedding api) and local vespa database
   Using concept: https://blog.vespa.ai/scaling-colpali-to-billions/
   
Requirements:
 - vespa (docker container)
 - Vision language model - e.g. Qwen2-VL-7B (via openai compatible api)

1. Clone repo
2. Create python venv or conda env
3. pip install -r requirements.txt
4. Download "Vespa CLI" package for your platform here -> https://github.com/vespa-engine/vespa/releases
   - copy bin/vespa to /usr/bin/vespa
   - Adjust permissions:
   ```
   sudo chmod +x /usr/bin/vespa && sudo chmod 755 /usr/bin/vespa
   ```
6. Start vespa local docker container
   ```
   docker run --detach --name vespa --hostname my-vespa-container --publish 8080:8080 --publish 19071:19071 vespaengine/vespa:latest
   ```
7. Start ColQwen2 via infinity embedding api like (only -merged versions work with infinity):
   ```
   infinity_emb v2 --device cuda --no-bettertransformer --batch-size 16 --dtype float16 --model-id vidore/colqwen2-v1.0-merged --served-model-name colqwen2 --api-key sk-1111
   ```  
   
8. Deploy vespa application schema to vespa instance (localhost)
   ```
   python deploy_vespa_app_local.py --vespa_application_name MyApplicationName
   ```

9. Embed and feed pdf files from a folder to vespa
   ```
   python feed-vespa_colqwen2-api.py --application_name MyApplicationName --vespa_schema_name pdf_page --pdf_folder /path/to/my/pdf/files/
   ```

10. Use vespa.py to test retrieval. Adjust "queries = [...]" in file as needed.
   ```
   python vespa_query-test.py
   ```

11. Import "openwebui_visual-retrieval-function.py" into Open-Webui as function
    - Configure Valves (Vespa DB host, etc.)
    
    - Currently written to work with the Colqwen infinity api behind litellm with a config like described here (https://github.com/BerriAI/litellm/issues/6525#issuecomment-2449697206)
    Otherwise "extra_body: { "modality": "image/text" }" is needed.
 
    litellm api cfg:
    ```
      - model_name: colqwen2
        litellm_params:
          model: "openai/vidore/colqwen2-v1.0-merged"
          api_base: "http://infinity-inference:7997"
          api_key: "sk-1111"
          extra_body: { "modality": "image" }
          model_info:
            id: "1"
            mode: "embedding"
        
      - model_name: colqwen2-text
        litellm_params:
          model: "openai/vidore/colqwen2-v1.0-merged"
          api_base: "http://infinity-inference:7997"
          api_key: "sk-1111"
          extra_body: { "modality": "text" }
          model_info:
            id: "1"
            mode: "embedding"
    ```

#### Thx to: 
   - Vespa ! (https://github.com/vespa-engine/sample-apps/tree/master/visual-retrieval-colpali)
   - Infinity (https://github.com/michaelfeil/infinity/)
   - Open-Webui https://github.com/open-webui/open-webui