1. **使用GlobalInstruction字段**：
   - 适用场景：输出语言要求适用于所有代理
   - 修改位置：在llmagent.Config中添加GlobalInstruction字段
   - 示例代码：`GlobalInstruction: "You MUST respond in the same language as the user's query, either Chinese or English."`

2. **使用InstructionProvider**：
   - 适用场景：需要动态生成提示词，或根据上下文调整
   - 修改位置：在llmagent.Config中添加InstructionProvider字段
   - 示例代码：
     ```go
     InstructionProvider: func(ctx agent.ReadonlyContext) (string, error) {
         return "Your SOLE purpose is to answer questions about the current time and weather in a specific city. You MUST refuse to answer any questions unrelated to time or weather. You MUST respond in the same language as the user's query, either Chinese or English.", nil
     },
     ```

3. **使用GlobalInstructionProvider**：
   - 适用场景：全局动态生成提示词
   - 修改位置：在llmagent.Config中添加GlobalInstructionProvider字段
   - 示例代码：
     ```go
     GlobalInstructionProvider: func(ctx agent.ReadonlyContext) (string, error) {
         return "You MUST respond in the same language as the user's query, either Chinese or English.", nil
     },
     ```

4. **使用BeforeModelCallback**：
   - 适用场景：需要在发送请求前动态修改提示词，或与其他逻辑结合
   - 修改位置：在llmagent.Config中添加BeforeModelCallbacks字段
   - 示例代码：
     ```go
     BeforeModelCallbacks: []llmagent.BeforeModelCallback{
         func(ctx agent.CallbackContext, llmRequest *model.LLMRequest) (*model.LLMResponse, error) {
             // 在现有SystemInstruction基础上添加输出语言要求
             if llmRequest.Config != nil && llmRequest.Config.SystemInstruction != nil {
                 // 读取现有system instruction内容
                 var existingInstruction string
                 for _, part := range llmRequest.Config.SystemInstruction.Parts {
                     if text, ok := part.(*genai.TextPart); ok {
                         existingInstruction += text.Text
                     }
                 }
                 // 添加输出语言要求
                 newInstruction := existingInstruction + "\nYou MUST respond in the same language as the user's query, either Chinese or English."
                 llmRequest.Config.SystemInstruction = genai.NewContentFromText(newInstruction, genai.RoleSystem)
             }
             return nil, nil
         },
     },
     ```

5. **修改现有Instruction字段（优化版）**：
   - 适用场景：简单直接，适合快速修改
   - 修改位置：现有的Instruction字段
   - 优化方式：将提示词拆分为多行，使结构更清晰
   - 示例代码：
     ```go
     Instruction: `Your SOLE purpose is to answer questions about the current time and weather in a specific city.
You MUST refuse to answer any questions unrelated to time or weather.
You MUST respond in the same language as the user's query, either Chinese or English.`,
     ```

