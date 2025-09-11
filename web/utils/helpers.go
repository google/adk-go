package utils

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/url"
	"reflect"
	"strconv"
	"strings"
	"time"
)

const errMsgRequiredMissing = "required parameter is missing"
const errMsgMinValueConstraint = "provided parameter is not respecting minimum value constraint"
const errMsgMaxValueConstraint = "provided parameter is not respecting maximum value constraint"

// IsZeroValue checks if the val is the zero-ed value.
func IsZeroValue(val interface{}) bool {
	return val == nil || reflect.DeepEqual(val, reflect.Zero(reflect.TypeOf(val)).Interface())
}

// EncodeJSONResponse uses the json encoder to write an interface to the http response with an optional status code
func EncodeJSONResponse(i interface{}, status int, w http.ResponseWriter) error {
	wHeader := w.Header()
	wHeader.Set("Content-Type", "application/json; charset=UTF-8")

	w.WriteHeader(status)

	if i != nil {
		return json.NewEncoder(w).Encode(i)
	}

	return nil
}

// parseTime will parses a string parameter into a time.Time using the RFC3339 format
func parseTime(param string) (time.Time, error) {
	if param == "" {
		return time.Time{}, nil
	}
	return time.Parse(time.RFC3339, param)
}

type Number interface {
	~int32 | ~int64 | ~float32 | ~float64
}

type ParseString[T Number | string | bool] func(v string) (T, error)

// parseFloat64 parses a string parameter to an float64.
func parseFloat64(param string) (float64, error) {
	if param == "" {
		return 0, nil
	}

	return strconv.ParseFloat(param, 64)
}

// parseFloat32 parses a string parameter to an float32.
func parseFloat32(param string) (float32, error) {
	if param == "" {
		return 0, nil
	}

	v, err := strconv.ParseFloat(param, 32)
	return float32(v), err
}

// parseInt64 parses a string parameter to an int64.
func parseInt64(param string) (int64, error) {
	if param == "" {
		return 0, nil
	}

	return strconv.ParseInt(param, 10, 64)
}

// parseInt32 parses a string parameter to an int32.
func parseInt32(param string) (int32, error) {
	if param == "" {
		return 0, nil
	}

	val, err := strconv.ParseInt(param, 10, 32)
	return int32(val), err
}

// parseBool parses a string parameter to an bool.
func parseBool(param string) (bool, error) {
	if param == "" {
		return false, nil
	}

	return strconv.ParseBool(param)
}

type Operation[T Number | string | bool] func(actual string) (T, bool, error)

func WithRequire[T Number | string | bool](parse ParseString[T]) Operation[T] {
	var empty T
	return func(actual string) (T, bool, error) {
		if actual == "" {
			return empty, false, errors.New(errMsgRequiredMissing)
		}

		v, err := parse(actual)
		return v, false, err
	}
}

func WithDefaultOrParse[T Number | string | bool](def T, parse ParseString[T]) Operation[T] {
	return func(actual string) (T, bool, error) {
		if actual == "" {
			return def, true, nil
		}

		v, err := parse(actual)
		return v, false, err
	}
}

func WithParse[T Number | string | bool](parse ParseString[T]) Operation[T] {
	return func(actual string) (T, bool, error) {
		v, err := parse(actual)
		return v, false, err
	}
}

type Constraint[T Number | string | bool] func(actual T) error

func WithMinimum[T Number](expected T) Constraint[T] {
	return func(actual T) error {
		if actual < expected {
			return errors.New(errMsgMinValueConstraint)
		}

		return nil
	}
}

func WithMaximum[T Number](expected T) Constraint[T] {
	return func(actual T) error {
		if actual > expected {
			return errors.New(errMsgMaxValueConstraint)
		}

		return nil
	}
}

// parseNumericParameter parses a numeric parameter to its respective type.
func parseNumericParameter[T Number](param string, fn Operation[T], checks ...Constraint[T]) (T, error) {
	v, ok, err := fn(param)
	if err != nil {
		return 0, err
	}

	if !ok {
		for _, check := range checks {
			if err := check(v); err != nil {
				return 0, err
			}
		}
	}

	return v, nil
}

// parseBoolParameter parses a string parameter to a bool
func parseBoolParameter(param string, fn Operation[bool]) (bool, error) {
	v, _, err := fn(param)
	return v, err
}

// parseNumericArrayParameter parses a string parameter containing array of values to its respective type.
func parseNumericArrayParameter[T Number](param, delim string, required bool, fn Operation[T], checks ...Constraint[T]) ([]T, error) {
	if param == "" {
		if required {
			return nil, errors.New(errMsgRequiredMissing)
		}

		return nil, nil
	}

	str := strings.Split(param, delim)
	values := make([]T, len(str))

	for i, s := range str {
		v, ok, err := fn(s)
		if err != nil {
			return nil, err
		}

		if !ok {
			for _, check := range checks {
				if err := check(v); err != nil {
					return nil, err
				}
			}
		}

		values[i] = v
	}

	return values, nil
}

// parseQuery parses query parameters and returns an error if any malformed value pairs are encountered.
func parseQuery(rawQuery string) (url.Values, error) {
	return url.ParseQuery(rawQuery)
}
