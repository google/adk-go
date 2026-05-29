package skilltoolset

import (
	"reflect"
	"strings"
	"testing"

	"google.golang.org/adk/tool/skilltoolset/internal/skilltool"
)

// TestDefaultInstructionMatchesLoadSkillSchema guards against the default
// system instruction documenting a parameter name for the load_skill tool that
// does not match its input schema. The instruction previously told the model to
// call load_skill with skill_name="<SKILL_NAME>", but LoadSkillArgs marshals its
// field as "name", so the first call was rejected for an unknown field.
func TestDefaultInstructionMatchesLoadSkillSchema(t *testing.T) {
	field, ok := reflect.TypeOf(skilltool.LoadSkillArgs{}).FieldByName("Name")
	if !ok {
		t.Fatal("LoadSkillArgs has no Name field")
	}
	paramName := strings.Split(field.Tag.Get("json"), ",")[0]
	if paramName == "" {
		t.Fatal("LoadSkillArgs.Name has no json tag")
	}

	if !strings.Contains(defaultSkillSystemInstruction, "`load_skill` tool with `"+paramName+"=") {
		t.Errorf("default instruction should document load_skill with %q=, got:\n%s", paramName, defaultSkillSystemInstruction)
	}
	if strings.Contains(defaultSkillSystemInstruction, "`load_skill` tool with `skill_name=") {
		t.Error("default instruction still documents skill_name= for load_skill, which is not a valid load_skill parameter")
	}
}
