function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

export function createDefaultLoopConfig() {
  return {
    max_iterations: 3,
    terminate_when: {
      mode: 'key_truthy',
      key: 'done',
      value: true
    },
    controller_layout: {
      input_position: { x: 40, y: 220 },
      output_position: { x: 980, y: 220 }
    },
    controller: {
      termination_mode: 'key_rule',
      terminate_condition_prompt: '',
      terminate_expression: '',
      model_settings: {
        model_name: '',
        base_url: ''
      }
    },
    subgraph: {
      nodes: [
        {
          id: 'loop_step_1',
          type: 'agent',
          label: 'Loop Step',
          position: { x: 260, y: 180 },
          config: {
            instructions: 'Review the current state and decide whether the task is complete.',
            prompt_template: 'Current state: {message}',
            pull_keys: { message: 'Loop input' },
            push_keys: { message: 'Loop output', done: 'Stop flag' },
            behavior_rules: [],
            knowledge: [],
            tools: []
          }
        }
      ],
      edges: []
    },
    controller_inputs: [
      {
        id: 'controller_in_1',
        target: 'loop_step_1',
        mapping: { message: 'Loop input' }
      }
    ],
    controller_outputs: [
      {
        id: 'controller_out_1',
        source: 'loop_step_1',
        mapping: { message: 'Loop output', done: 'Stop flag' }
      }
    ]
  };
}

export function normalizeLoopConfig(rawConfig = {}) {
  const config = clone(rawConfig || {});
  if (config.subgraph && Array.isArray(config.subgraph.nodes) && Array.isArray(config.subgraph.edges)) {
    return {
      max_iterations: Number(config.max_iterations || 3),
      terminate_when: {
        mode: config.terminate_when?.mode || 'never',
        key: config.terminate_when?.key || '',
        value: config.terminate_when?.value ?? true
      },
      controller_layout: {
        input_position: {
          x: Number(config.controller_layout?.input_position?.x ?? 40),
          y: Number(config.controller_layout?.input_position?.y ?? 220)
        },
        output_position: {
          x: Number(config.controller_layout?.output_position?.x ?? 980),
          y: Number(config.controller_layout?.output_position?.y ?? 220)
        }
      },
      controller: {
        termination_mode: config.controller?.termination_mode || 'key_rule',
        terminate_condition_prompt: config.controller?.terminate_condition_prompt || '',
        terminate_expression: config.controller?.terminate_expression || '',
        model_settings: {
          model_name: config.controller?.model_settings?.model_name || '',
          base_url: config.controller?.model_settings?.base_url || ''
        }
      },
      subgraph: {
        nodes: clone(config.subgraph.nodes || []),
        edges: clone(config.subgraph.edges || [])
      },
      controller_inputs: clone(config.controller_inputs || []),
      controller_outputs: clone(config.controller_outputs || []),
      attributes: clone(config.attributes || {})
    };
  }

  if (config.body) {
    const bodyNodeId = 'loop_step_1';
    return {
      max_iterations: Number(config.max_iterations || 3),
      terminate_when: {
        mode: config.terminate_when?.mode || 'never',
        key: config.terminate_when?.key || '',
        value: config.terminate_when?.value ?? true
      },
      controller_layout: {
        input_position: { x: 40, y: 220 },
        output_position: { x: 980, y: 220 }
      },
      controller: {
        termination_mode: 'key_rule',
        terminate_condition_prompt: '',
        terminate_expression: '',
        model_settings: {
          model_name: '',
          base_url: ''
        }
      },
      subgraph: {
        nodes: [
          {
            id: bodyNodeId,
            type: config.body.type || 'agent',
            label: 'Loop Step',
            position: { x: 260, y: 180 },
            config: {
              ...clone(config.body),
              pull_keys: clone(config.body.pull_keys || config.body.input_mapping || {}),
              push_keys: clone(config.body.push_keys || config.body.output_mapping || {})
            }
          }
        ],
        edges: []
      },
      controller_inputs: [
        {
          id: 'controller_in_1',
          target: bodyNodeId,
          mapping: clone(config.body.input_mapping || config.body.pull_keys || { message: 'Loop input' })
        }
      ],
      controller_outputs: [
        {
          id: 'controller_out_1',
          source: bodyNodeId,
          mapping: clone(config.body.output_mapping || config.body.push_keys || { message: 'Loop output' })
        }
      ],
      attributes: clone(config.attributes || {})
    };
  }

  return createDefaultLoopConfig();
}

export function createDemoDocument() {
  return {
    id: 'demo_clawcanvas',
    name: 'ClawCanvas Demo Skill',
    description: 'A MASFactory-based skill design workflow.',
    inputs: {
      query: 'Design a skill studio based on MASFactory.'
    },
    attributes: {},
    key_descriptions: {
      query: 'Original user request',
      analysis: 'Structured analysis',
      draft: 'Interim loop draft',
      answer: 'Final design brief'
    },
    manifest: {
      name: 'clawcanvas_demo_skill',
      version: '0.1.0',
      description: 'Demo Skill package.',
      tags: ['demo', 'workflow', 'skill'],
      tools: [
        {
          name: 'web_search',
          binding: 'future',
          description: 'Reserved tool binding metadata'
        }
      ],
      knowledge: [
        {
          title: 'Skill Definition',
          text: 'Skill = tools + domain knowledge + behavior rules.'
        }
      ],
      behavior: {
        style: 'structured',
        rules: ['Explain the workflow clearly.', 'Keep the output actionable.']
      }
    },
    nodes: [
      {
        id: 'start',
        type: 'start',
        label: 'Start',
        position: { x: 80, y: 240 },
        config: {}
      },
      {
        id: 'researcher',
        type: 'agent',
        label: 'Researcher',
        position: { x: 340, y: 160 },
        config: {
          instructions: 'Analyze the user request and extract the workflow goals.',
          prompt_template: 'User request: {query}',
          pull_keys: { query: 'Original user request' },
          push_keys: { analysis: 'Structured analysis' },
          behavior_rules: ['Focus on requirements.', 'List assumptions briefly.'],
          knowledge: []
        }
      },
      {
        id: 'review_loop',
        type: 'loop',
        label: 'Review Loop',
        position: { x: 700, y: 180 },
        config: {
          max_iterations: 3,
          terminate_when: { mode: 'key_truthy', key: 'done', value: true },
          controller_layout: {
            input_position: { x: 40, y: 220 },
            output_position: { x: 980, y: 220 }
          },
          controller: {
            termination_mode: 'key_rule',
            terminate_condition_prompt: '',
            terminate_expression: '',
            model_settings: {
              model_name: ''
            }
          },
          subgraph: {
            nodes: [
              {
                id: 'draft_writer',
                type: 'agent',
                label: 'Draft Writer',
                position: { x: 240, y: 120 },
                config: {
                  instructions: 'Turn the analysis into a draft design brief.',
                  prompt_template: 'Analysis: {analysis}',
                  pull_keys: { analysis: 'Structured analysis' },
                  push_keys: { draft: 'Draft design brief' },
                  behavior_rules: [],
                  knowledge: [],
                  tools: []
                }
              },
              {
                id: 'draft_reviewer',
                type: 'custom',
                label: 'Draft Reviewer',
                position: { x: 560, y: 220 },
                config: {
                  mode: 'set',
                  pull_keys: { draft: 'Draft design brief' },
                  push_keys: { answer: 'Final design brief', done: 'Stop flag' },
                  static_outputs: {
                    answer: 'Reviewed: {draft}',
                    done: 'true'
                  },
                  pick_keys: {}
                }
              }
            ],
            edges: [
              {
                id: 'inner_edge_1',
                source: 'draft_writer',
                target: 'draft_reviewer',
                mapping: { draft: 'Draft design brief' }
              }
            ]
          },
          controller_inputs: [
            {
              id: 'controller_in_1',
              target: 'draft_writer',
              mapping: { analysis: 'Structured analysis' }
            }
          ],
          controller_outputs: [
            {
              id: 'controller_out_1',
              source: 'draft_reviewer',
              mapping: { answer: 'Final design brief', done: 'Stop flag' }
            }
          ]
        }
      },
      {
        id: 'end',
        type: 'end',
        label: 'End',
        position: { x: 1080, y: 220 },
        config: {}
      }
    ],
    edges: [
      {
        id: 'edge_1',
        source: 'start',
        target: 'researcher',
        mapping: { query: 'User request' }
      },
      {
        id: 'edge_2',
        source: 'researcher',
        target: 'review_loop',
        mapping: { analysis: 'Analysis result' }
      },
      {
        id: 'edge_3',
        source: 'review_loop',
        target: 'end',
        mapping: { answer: 'Final answer' }
      }
    ]
  };
}

export function createEmptyDocument() {
  return {
    id: `skill_${Date.now()}`,
    name: 'Untitled Skill',
    description: '',
    inputs: {
      query: ''
    },
    attributes: {},
    key_descriptions: {
      query: 'Original user request',
      answer: 'Final answer'
    },
    manifest: {
      name: 'untitled-skill',
      version: '0.1.0',
      description: '',
      tags: [],
      tools: [],
      knowledge: [],
      behavior: {
        style: 'structured',
        rules: []
      }
    },
    nodes: [
      {
        id: 'start',
        type: 'start',
        label: 'Start',
        position: { x: 80, y: 220 },
        config: {}
      },
      {
        id: 'end',
        type: 'end',
        label: 'End',
        position: { x: 760, y: 220 },
        config: {}
      }
    ],
    edges: []
  };
}

export function nextNodeId(document, prefix) {
  const taken = new Set((document.nodes || []).map((node) => node.id));
  let index = 1;
  while (taken.has(`${prefix}_${index}`)) {
    index += 1;
  }
  return `${prefix}_${index}`;
}

export function buildNodeTemplate(type, id) {
  if (type === 'agent') {
    return {
      id,
      type: 'agent',
      label: `Reasoning ${id.split('_')[1] || id}`,
      position: { x: 240, y: 160 },
      config: {
        instructions: 'Describe this node behavior.',
        prompt_template: '{message}',
        pull_keys: { message: 'Input message' },
        push_keys: { message: 'Output message' },
        behavior_rules: [],
        knowledge: [],
        tools: []
      }
    };
  }

  if (type === 'custom') {
    return {
      id,
      type: 'custom',
      label: `Logic ${id.split('_')[1] || id}`,
      position: { x: 240, y: 220 },
      config: {
        mode: 'python',
        python_code: `def forward(input_dict, attributes):
    message = input_dict.get("message", "")
    return {
        "message": f"processed: {message}",
    }
`,
        templates: { message: 'Transformed: {message}' },
        static_outputs: {},
        pick_keys: { original_message: 'message' },
        pull_keys: { message: 'Input message' },
        push_keys: { message: 'Output message', original_message: 'Copied input message' }
      }
    };
  }

  if (type === 'loop') {
    return {
      id,
      type: 'loop',
      label: `Loop ${id.split('_')[1] || id}`,
      position: { x: 240, y: 280 },
      config: createDefaultLoopConfig()
    };
  }

  throw new Error(`Unsupported node template type: ${type}`);
}
