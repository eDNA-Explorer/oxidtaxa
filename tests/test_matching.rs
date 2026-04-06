mod common;

use common::load_json;
use oxidaxa::matching::int_match;
use std::collections::HashMap;

/// Load intMatch test cases from golden JSON.
#[derive(serde::Deserialize)]
struct IntMatchCase {
    x: Vec<i32>,
    y: Vec<i32>,
}

#[test]
fn test_int_match_basic() {
    let cases: HashMap<String, IntMatchCase> = load_json("s07_im_cases");
    let golden_basic: Vec<bool> = load_json("s07_im_basic");

    let case = cases.get("basic").unwrap();
    let result = int_match(&case.x, &case.y);
    assert_eq!(result, golden_basic, "basic intMatch mismatch");
}

#[test]
fn test_int_match_no_overlap() {
    let cases: HashMap<String, IntMatchCase> = load_json("s07_im_cases");
    let golden: Vec<bool> = load_json("s07_im_no_overlap");

    let case = cases.get("no_overlap").unwrap();
    let result = int_match(&case.x, &case.y);
    assert_eq!(result, golden, "no_overlap intMatch mismatch");
}

#[test]
fn test_int_match_all_match() {
    let cases: HashMap<String, IntMatchCase> = load_json("s07_im_cases");
    let golden: Vec<bool> = load_json("s07_im_all_match");

    let case = cases.get("all_match").unwrap();
    let result = int_match(&case.x, &case.y);
    assert_eq!(result, golden, "all_match intMatch mismatch");
}

#[test]
fn test_int_match_empty_x() {
    let cases: HashMap<String, IntMatchCase> = load_json("s07_im_cases");
    let golden: Vec<bool> = load_json("s07_im_empty_x");

    let case = cases.get("empty_x").unwrap();
    let result = int_match(&case.x, &case.y);
    assert_eq!(result, golden, "empty_x intMatch mismatch");
}

#[test]
fn test_int_match_empty_y() {
    let cases: HashMap<String, IntMatchCase> = load_json("s07_im_cases");
    let golden: Vec<bool> = load_json("s07_im_empty_y");

    let case = cases.get("empty_y").unwrap();
    let result = int_match(&case.x, &case.y);
    assert_eq!(result, golden, "empty_y intMatch mismatch");
}

#[test]
fn test_int_match_single() {
    let cases: HashMap<String, IntMatchCase> = load_json("s07_im_cases");
    let golden: Vec<bool> = load_json("s07_im_single");

    let case = cases.get("single").unwrap();
    let result = int_match(&case.x, &case.y);
    assert_eq!(result, golden, "single intMatch mismatch");
}

#[test]
fn test_int_match_large() {
    let cases: HashMap<String, IntMatchCase> = load_json("s07_im_cases");
    let golden: Vec<bool> = load_json("s07_im_large");

    let case = cases.get("large").unwrap();
    let result = int_match(&case.x, &case.y);
    assert_eq!(result, golden, "large intMatch mismatch");
}
