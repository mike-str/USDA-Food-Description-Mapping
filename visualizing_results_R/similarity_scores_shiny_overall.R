# app.R - Score‑density dashboard

library(shiny)
library(ggplot2)
library(dplyr)
library(tools)
library(stringr)

# ── helper: prettify file names ------------------------------------------------
format_label <- function(filename) {
  name  <- file_path_sans_ext(filename)
  parts <- unlist(strsplit(name, "_"))
  parts <- tolower(parts) |> tools::toTitleCase()
  parts[1] <- toupper(parts[1])
  paste(parts, collapse = " ")
}

# ── load all experiment CSVs ---------------------------------------------------
load_all_experiments <- function() {
  csv_dir <- file.path("..", "results", "csv_files")
  files   <- list.files(csv_dir, pattern = "\\.csv$", full.names = TRUE)
  df_list <- lapply(files, function(f) {
    df <- read.csv(f, stringsAsFactors = FALSE)
    df$source_file <- basename(f)
    df
  })
  bind_rows(df_list)
}

experiment_data <- load_all_experiments()

# ── mapping for dropdown -------------------------------------------------------
file_choices <- unique(experiment_data$source_file)
names(file_choices) <- sapply(file_choices, format_label)

nhanes_files <- c("nhanes_experiment_2.csv", "nhanes_experiment_4.csv")

# ── UI -------------------------------------------------------------------------
ui <- fluidPage(
  titlePanel("Score Density Plots by Experiment"),
  fluidRow(
    column(3,
           selectInput("experiment_select", "Select Experiment:",
                       choices = file_choices, width = "100%")),
    column(3,
           checkboxGroupInput("overlay_options", "Overlay Options:",
                              choices = c("Mean" = "mean", "Q1/Q3" = "q1q3"),
                              selected = character(0), inline = TRUE)),
    column(3,
           radioButtons("ignore_exact_match", "Ignore Exact Matches:",
                        choices = c("No", "Yes"), selected = "No", inline = TRUE)),
    column(3, uiOutput("custom_filter_ui"))
  ),
  fluidRow(column(12, textOutput("data_summary"), tags$hr())),
  plotOutput("plot_fuzzy",  height = "250px"),
  plotOutput("plot_tfidf",  height = "250px"),
  plotOutput("plot_embed",  height = "250px")
)

# ── SERVER ---------------------------------------------------------------------
server <- function(input, output, session) {
  
  # dynamic filter UI -----------------------------------------------------------
  output$custom_filter_ui <- renderUI({
    if (input$experiment_select %in% nhanes_files) {
      radioButtons("label_filter",
                   "Filter (0 not in DB, 1 in DB, 2 both, 3 overlay):",
                   choices = c("0", "1", "2", "3"), selected = "2", inline = TRUE)
    } else {
      radioButtons("correct_filter", "Prediction:",
                   choices = c("Both", "Both Overlay", "Correct", "Incorrect"),
                   selected = "Both", inline = TRUE)
    }
  })
  
  # main reactive dataset -------------------------------------------------------
  selected_data <- reactive({
    df <- experiment_data %>% filter(source_file == input$experiment_select)
    
    ## NHANES‑specific label logic ---------------------------------------------
    if (input$experiment_select %in% nhanes_files) {
      lbl <- input$label_filter
      if (lbl == "0") {
        df <- df %>% filter(label == 0)
      } else if (lbl == "1") {
        df <- df %>% filter(label == 1)
      } else if (lbl == "3" && "label" %in% names(df)) {
        df <- df %>% mutate(pred_status = ifelse(label == 1,
                                                 "In DB", "Not in DB"))
      }
    }
    
    ## remove exact matches if requested ---------------------------------------
    if (input$ignore_exact_match == "Yes") {
      df <- df %>% filter(input_desc != target_desc)
    }
    
    df
  })
  
  # experiment summary header ---------------------------------------------------
  output$data_summary <- renderText({
    full_df     <- experiment_data %>% filter(source_file == input$experiment_select)
    exact_count <- sum(full_df$input_desc == full_df$target_desc, na.rm = TRUE)
    paste0("Currently using: ", nrow(selected_data()),
           " rows   |   Total in experiment: ", nrow(full_df),
           "   |   Exact input‑target matches: ", exact_count)
  })
  
  # plotting helper -------------------------------------------------------------
  render_score_density <- function(score_col, match_col,
                                   method_name, subtitle_text,
                                   scale_max = 1) {
    
    df <- selected_data()
    
    # ── non‑NHANES Correct/Incorrect overlay & filtering ----------------------
    if (!(input$experiment_select %in% nhanes_files)) {
      if (match_col %in% names(df)) {
        is_correct <- df[[match_col]] == df$target_desc
        is_correct[is.na(is_correct)] <- FALSE
      } else {
        is_correct <- rep(FALSE, nrow(df))  # fail‑safe
      }
      
      if (input$correct_filter == "Correct") {
        df <- df[is_correct, ]
      } else if (input$correct_filter == "Incorrect") {
        df <- df[!is_correct, ]
      } else if (input$correct_filter == "Both Overlay") {
        df <- df %>% mutate(pred_status = ifelse(is_correct,
                                                 "Correct", "Incorrect"))
      }
    }
    
    # basic sanity check -------------------------------------------------------
    df <- df %>%
      filter(!is.na(.data[[score_col]]),
             .data[[score_col]] >= 0,
             .data[[score_col]] <= scale_max)
    
    if (nrow(df) < 5 || sd(df[[score_col]]) == 0) {
      return(
        ggplot() +
          labs(title    = paste(method_name, "Score Density"),
               subtitle = "Insufficient or constant data") +
          theme_void() +
          theme(plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
                plot.subtitle = element_text(hjust = 0.5))
      )
    }
    
    # shared objects -----------------------------------------------------------
    breaks       <- if (scale_max == 100) seq(0, 100, 10) else seq(0, 1, 0.1)
    palette_vals <- c("Correct"    = "#8BC34A",
                      "Incorrect"  = "#B22222",
                      "In DB"      = "#8BC34A",
                      "Not in DB"  = "#B22222")
    
    # should we draw overlay? --------------------------------------------------
    overlay_mode <- (
      input$experiment_select %in% nhanes_files &&
        !is.null(input$label_filter) &&
        input$label_filter == "3"
    ) ||
      (
        !(input$experiment_select %in% nhanes_files) &&
          !is.null(input$correct_filter) &&
          input$correct_filter == "Both Overlay"
      )
    
    if (overlay_mode && "pred_status" %in% names(df)) {
      # ── overlay plot ========================================================
      group_stats <- df %>%
        group_by(pred_status) %>%
        summarise(mean_val = mean(.data[[score_col]]),
                  q1_val   = quantile(.data[[score_col]], 0.25),
                  q3_val   = quantile(.data[[score_col]], 0.75),
                  .groups  = "drop")
      
      y_max <- max(
        sapply(split(df[[score_col]], df$pred_status),
               function(x) max(density(x, adjust = 1.25)$y))
      ) * 1.05
      
      p <- ggplot(df, aes(x = .data[[score_col]],
                          fill = pred_status,
                          colour = pred_status)) +
        geom_density(adjust = 1.25, alpha = 0.35, linewidth = 1) +
        scale_fill_manual(values = palette_vals) +
        scale_colour_manual(values = palette_vals) +
        scale_x_continuous(limits = c(0, scale_max), breaks = breaks,
                           expand = c(0, 0)) +
        scale_y_continuous(limits = c(0, y_max), expand = c(0, 0)) +
        labs(title = paste(method_name, "Score Density"),
             subtitle = subtitle_text, x = "Score", y = NULL) +
        theme_classic(base_size = 14) +
        theme(plot.title    = element_text(size = 16, face = "bold", hjust = 0.5),
              plot.subtitle = element_text(hjust = 0.5, size = 12),
              axis.text.y   = element_blank(), axis.ticks.y = element_blank(),
              legend.position = "none",
              plot.margin = margin(5, 15, 5, 15))
      
      # optional guides --------------------------------------------------------
      if ("mean" %in% input$overlay_options) {
        mean_labels <- group_stats %>%
          mutate(label_text = paste0(pred_status, ":   ",
                                     sprintf("%.3f", mean_val)),
                 x = 0.02 * scale_max,
                 y_base = y_max * 0.95,
                 y_offset = 0.06 * y_max,
                 y = y_base - (row_number() - 1) * y_offset -
                   ifelse(pred_status %in% c("Incorrect", "Not in DB"),
                          0.01 * y_max, 0))
        
        p <- p +
          geom_vline(data = group_stats,
                     aes(xintercept = mean_val, colour = pred_status),
                     linewidth = 1) +
          geom_text(data = mean_labels,
                    aes(x = x, y = y, label = label_text, colour = pred_status),
                    hjust = 0, vjust = 1, size = 4, show.legend = FALSE)
      }
      
      if ("q1q3" %in% input$overlay_options) {
        p <- p +
          geom_vline(data = group_stats,
                     aes(xintercept = q1_val, colour = pred_status),
                     linetype = "dashed") +
          geom_vline(data = group_stats,
                     aes(xintercept = q3_val, colour = pred_status),
                     linetype = "dashed")
      }
      
    } else {
      # ── single‑distribution plot ============================================
      density_data <- density(df[[score_col]], adjust = 1.25)
      y_max        <- max(density_data$y) * 1.05
      mean_val     <- mean(df[[score_col]])
      q1_val       <- quantile(df[[score_col]], 0.25)
      q3_val       <- quantile(df[[score_col]], 0.75)
      
      p <- ggplot(df, aes(x = .data[[score_col]])) +
        geom_density(adjust = 1.25,
                     fill = "grey85", colour = "black", alpha = 0.8) +
        scale_x_continuous(limits = c(0, scale_max), breaks = breaks,
                           expand = c(0, 0)) +
        scale_y_continuous(limits = c(0, y_max), expand = c(0, 0)) +
        labs(title = paste(method_name, "Score Density"),
             subtitle = subtitle_text, x = "Score", y = NULL) +
        theme_classic(base_size = 14) +
        theme(plot.title    = element_text(size = 16, face = "bold", hjust = 0.5),
              plot.subtitle = element_text(hjust = 0.5, size = 12),
              axis.text.y   = element_blank(), axis.ticks.y = element_blank(),
              plot.margin   = margin(5, 15, 5, 15))
      
      if ("mean" %in% input$overlay_options) {
        p <- p +
          geom_vline(xintercept = mean_val, colour = "firebrick", linewidth = 1) +
          annotate("text",
                   x = 0.02 * scale_max, y = y_max * 0.95,
                   label = sprintf("Mean: %.3f", mean_val),
                   hjust = 0, vjust = 1, size = 4, colour = "firebrick")
      }
      
      if ("q1q3" %in% input$overlay_options) {
        p <- p +
          geom_vline(xintercept = q1_val, colour = "grey50",
                     linetype = "dashed") +
          geom_vline(xintercept = q3_val, colour = "grey50",
                     linetype = "dashed")
      }
    }
    
    p
  }
  
  # ── three plots -------------------------------------------------------------
  output$plot_fuzzy <- renderPlot({
    render_score_density("score_fuzzy", "match_fuzzy",
                         "Fuzzy", "Levenshtein Similarity",
                         scale_max = 100)
  })
  
  output$plot_tfidf <- renderPlot({
    render_score_density("score_tfidf", "match_tfidf",
                         "TF-IDF", "Cosine Similarity")
  })
  
  output$plot_embed <- renderPlot({
    render_score_density("score_embed", "match_embed",
                         "Embed", "Cosine Similarity")
  })
}

shinyApp(ui, server)
